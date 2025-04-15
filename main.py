import argparse
import copy
import os
import re
# from time import time
import time
import torch
import numpy as np
from tqdm import tqdm
from datasets import Dataset
import random
import pandas as pd
from torch import cuda
from torchkge.models import TransEModel
from torchkge.sampling import BernoulliNegativeSampler
from torchkge.utils import MarginLoss, DataLoader
from torchkge.utils.datasets import load_wn18rr
from torchkge.evaluation import LinkPredictionEvaluator
from torchkge.data_structures import KnowledgeGraph
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, LlamaTokenizer
# Load Pretrained LLM (e.g., Llama 2 or GPT-3.5)
import ollama
from utils import set_seed

kg_train, kg_val, kg_test = load_wn18rr("data/WN18RR")
print(kg_train.head_idx.shape, kg_val.tail_idx.shape, kg_test.relations.shape)
kg_train_df = pd.DataFrame({'from': kg_train.head_idx, 'rel': kg_train.relations, 'to': kg_train.tail_idx})


def generate_candidate_links(kg, num_candidates=100):
    """Generate initial candidate links using both random selection and LLM guidance."""
    population = list()
    head, tail, relation = kg.head_idx.tolist(), kg.tail_idx.tolist(), kg.relations.tolist()
    num_triples = 1000

    # Random Generation
    for _ in range(num_candidates):
        # h = random.sample(head, num_triples)
        # t = random.sample(tail, num_triples)
        # r = random.sample(relation, num_triples)
        sample_idx = random.sample(range(len(head)), num_triples)
        candidate = list()
        for i in range(num_triples):
            h = head[sample_idx[i]]
            t = tail[sample_idx[i]]
            r = relation[sample_idx[i]]
            candidate.append((h, r, t))
        population.append(candidate)

    return population

def evaluate_fitness(population, kg, e_model, batch_size, optimizer, criterion, device="cuda"):
    """Evaluate fitness using LLM plausibility and graph consistency scores."""
    fitness_scores = []
    for candidate in population:
        candidate_e_model = copy.deepcopy(e_model)
        
        df = pd.DataFrame(candidate, columns=['from', 'rel', 'to'])
        # kg_candidate = KnowledgeGraph(kg={'heads': kg.head_idx, 
        #                                   'tails': kg.tail_idx, 
        #                                   'relations': kg.relations}, 
        #                               ent2ix=kg.ent2ix, 
        #                               rel2ix=kg.rel2ix, 
        #                               dict_of_heads=kg.dict_of_heads, 
        #                               dict_of_tails=kg.dict_of_tails, 
        #                               dict_of_rels=kg.dict_of_rels)
        kg_candidate = KnowledgeGraph(kg={'heads': torch.concat([kg.head_idx, torch.tensor(df['from']).long()]), 
                                          'tails': torch.concat([kg.tail_idx, torch.tensor(df['to']).long()]), 
                                          'relations': torch.concat([kg.relations, torch.tensor(df['rel']).long()])}, 
                                      ent2ix=kg.ent2ix, rel2ix=kg.rel2ix, 
                                      dict_of_heads=kg.dict_of_heads, 
                                      dict_of_tails=kg.dict_of_tails, 
                                      dict_of_rels=kg.dict_of_rels)
        # kg_candidate = KnowledgeGraph(kg={'heads': torch.tensor(df['from']).long(), 
        #                                   'tails': torch.tensor(df['to']).long(), 
        #                                   'relations': torch.tensor(df['rel']).long()}, 
        #                               ent2ix=kg.ent2ix, rel2ix=kg.rel2ix, 
        #                               dict_of_heads=kg.dict_of_heads, 
        #                               dict_of_tails=kg.dict_of_tails, 
        #                               dict_of_rels=kg.dict_of_rels)
        # kg_candidate = KnowledgeGraph(kg_train_df)
        dataloader_candidate = DataLoader(kg_candidate, batch_size=batch_size)
        # Initialize sampler with correct number of relations
        sampler = BernoulliNegativeSampler(kg_candidate)
        
        optimizer = torch.optim.Adam(candidate_e_model.parameters(), lr=0.0001)

        candidate_e_model = trainer(candidate_e_model, dataloader_candidate, criterion, optimizer, sampler, num_epochs=10, device=device)

        candidate_e_model.eval()
        # Evaluate on validation set
        evaluator = LinkPredictionEvaluator(candidate_e_model, kg_val)
        evaluator.evaluate(b_size=batch_size, verbose=True)
        evaluator.print_results()
        mrr = evaluator.mrr()[0]
        fitness_scores.append(mrr)
    
    return fitness_scores

def selection(candidates, fitness_scores, num_selected=50):
    """Select top candidates based on fitness scores."""
    sorted_indices = np.argsort(fitness_scores)[-num_selected:]
    return [candidates[i] for i in sorted_indices], [fitness_scores[i] for i in sorted_indices]

def crossover(parents, num_offspring=50):
    """Perform crossover between parent link sets."""
    offsprings = list()
    for i in range(num_offspring):
        p1, p2 = random.sample(parents, 2)
        idx = random.randint(0, len(p1) - 1)
        offspring = p1[:idx] + p2[idx:]
        offsprings.append(offspring)
    return (offsprings)


def extract_triples(response):
    # Use regex to find all triples in the format (x, y, z)
    triple_pattern = r'\(\s*([a-zA-Z0-9_]+|"[^"]+")\s*,\s*([a-zA-Z0-9_]+|"[^"]+")\s*,\s*([a-zA-Z0-9_]+|"[^"]+")\s*\)'
    triples = re.findall(triple_pattern, response)
    
    # Filter and format the triples, keeping only those with integer values
    formatted_triples = []
    for x, y, z in triples:
        # Remove quotes from strings (e.g., "food" -> food)
        x = x.strip('"')
        y = y.strip('"')
        z = z.strip('"')
        
        # Check if all values (x, y, z) are integers
        if x.isdigit() and y.isdigit() and z.isdigit():
            formatted_triples.append((int(x), int(y), int(z)))
    
    return formatted_triples

def parse_triples_from_response(response):
    # Use regex to extract triples in the form (head, relation, tail)
    triple_pattern = r'\((\d+),\s*(\d+),\s*(\d+)\)'
    triples = re.findall(triple_pattern, response)
    
    # Convert the list of string tuples into a list of integer tuples
    triples = [(int(h), int(r), int(t)) for h, r, t in triples]

    # Remove duplicates
    triples = list(set(triples))
    
    return triples


def mutate(population, llm_model, mutation_rate=0.5):
    """Mutate candidate triples using LLM guidance."""
    mutated = list()
    # entities = kg_train["entities"]
    # entities = kg_train.head_idx.tolist() + kg_train.tail_idx.tolist()

    # relations = kg_train.relations.tolist()

    for candidate in population:
        size = len(candidate)
        idx = np.random.choice(len(kg_train.head_idx), size=100, replace=False)
        if random.random() < mutation_rate:
            # head, relation, tail = candidate
            head = kg_train.head_idx[idx]
            relation = kg_train.relations[idx]
            tail = kg_train.tail_idx[idx]

            # prompt = "Given the following sub knowledge graph:\n["
            # for i in range(100):
            #     prompt += f" ({head[i]}, {relation[i]}, {tail[i]}) "
            #     if i != 99:
            #         prompt += f","
            prompt = "Input Subgraph (100 triples):"
            for i, triple in enumerate(candidate):
                h, r, t = triple
                prompt += f"{i+1}. ({h}, {r}, {t})\n"
            prompt += f"Mutation Rate: 10% (mutate 10 triples and retain other {size-10} triples)\nGenerate a mutated subgraph of exactly {size} triples.  Respond only with the triples, without any additional text or explanations."

            response = ollama.chat(
                model=llm_model,
                messages=[
                    {"role": "system", "content": f"You are a mutation operator in an evolutionary algorithm for optimizing sub knowledge graphs based on the full knowledge graph:\n{kg_train_df.head(100)}\nYour task is to generate a mutated subgraph of exactly {len(candidate)} triples in the form (head, relation, tail). The input will be a subgraph of {len(candidate)} triples, and you must mutate it by introducing changes based on the mutation rate. The mutations should maintain the semantic consistency of the knowledge graph while introducing diversity. Respond only with the mutated subgraph of exactly {len(candidate)} triples, without any additional text or explanations."},
                    {"role": "user", "content": prompt}
                ],
                # keep_alive=
            )
            print(response['message']['content'])
            formatted_triples = parse_triples_from_response(response['message']['content'])
            # remove relation or head or tail that is not in the original graph
            formatted_triples = [triple for triple in formatted_triples if 0<=triple[0] < kg_train.n_ent and 0<=triple[1] < kg_train.n_rel and 0<=triple[2] < kg_train.n_ent]
            print(formatted_triples)
            for i in range(len(candidate) - len(formatted_triples)):
                rand_idx = random.randint(0, len(kg_train.head_idx) - 1)
                formatted_triples.append((kg_train.head_idx[rand_idx].item(), kg_train.relations[rand_idx].item(), kg_train.tail_idx[rand_idx].item()))
            mutated.append(formatted_triples)
           
        else:
            mutated.append(candidate)
    return (mutated)

def evolutionary_algorithm(kg, e_model, llm_model, batch_size, optimizer, criterion, generations=10, pop_size=100, mutation_rate=0.5, device="cuda"):
    """Run the evolutionary algorithm to predict missing links in WN18RR."""
    population = generate_candidate_links(kg, num_candidates=pop_size)
    # e_model_clone = copy.deepcopy(e_model)
    population_fitness = evaluate_fitness(population, kg, e_model, batch_size, optimizer, criterion, device=device)
    offspring = None
    offspring_fitness = []
    for gen in range(generations):
        
        if offspring is not None:
            offspring_fitness = evaluate_fitness(offspring, kg, e_model, batch_size, optimizer, criterion, device=device)
            population = population + offspring
            population_fitness = population_fitness + offspring_fitness
        population, population_fitness = selection(population, population_fitness, num_selected=pop_size)
        offspring = crossover(population, num_offspring=pop_size//1)
        offspring = mutate(offspring, llm_model, mutation_rate=mutation_rate)
        print(f"Generation {gen+1}: Best fitness = {max(population_fitness):.4f}")
    
    return population, population_fitness

def TransETrainer(model, dataloader, criterion, optimizer, sampler, num_epochs=100, device="cuda"):
    """Train TransE model on WN18RR dataset."""
    model.train()
    running_loss = 0.0
    iterator = tqdm(range(num_epochs), unit='epoch')
    
    for epoch in iterator:
        running_loss = 0.0
        for batch in dataloader:
            head, tail, relation = batch[0], batch[1], batch[2]
            head, tail, relation = head.to(device), tail.to(device), relation.to(device)
            
            # Generate negative samples on CPU
            n_h, n_t = sampler.corrupt_batch(head, tail, relation)
            
            # Move negative samples to GPU
            n_h, n_t = n_h.to(device), n_t.to(device)
            # print(n_h.device, n_t.device, relation.device, head.device, tail.device)
            # Forward pass
            optimizer.zero_grad()
            pos, neg = model(head, tail, relation, n_h, n_t)
            loss = criterion(pos, neg)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # Print progress
        avg_loss = running_loss / len(dataloader)
        iterator.set_description(
            'Epoch {} | mean loss: {:.5f}'.format(epoch + 1, avg_loss)
        )
        
        # Normalize embeddings after each epoch
        model.normalize_parameters()
    
    return model

    
def trainer(model, dataloader, criterion, optimizer, sampler, num_epochs=100, device="cuda"):
    """Train TransE model on WN18RR dataset."""
    # Setup the trainer
    TransETrainer(model, dataloader, criterion, optimizer, sampler, num_epochs=num_epochs, device=device)

    return model

def evaluate_on_test_set(model, kg_test):
    """Evaluate predicted links using MRR and Hits@10 metrics."""
    # Prepare test triples for evaluation
    # test_triples = torch.tensor(kg_test["triples"], dtype=torch.long)
    
    # ranks = evaluate_performance(model, predicted_links, test_triples, kg_train["triples"])
    
    # # Compute MRR and Hits@10
    # mrr = mrr_score(ranks)
    # hits_at_10 = hits_at_n_score(ranks, n=10)

    evaluator = LinkPredictionEvaluator(model, kg_test)
    evaluator.evaluate(b_size=256, verbose=False)
    evaluator.print_results()
    test_mrr = evaluator.mrr()[0]
    test_hits_at_10 = evaluator.hit_at_k(k=10)[0]
    
    print(f"MRR: {test_mrr:.4f}")
    print(f"Hits@10: {test_hits_at_10:.4f}")
    
    return test_mrr, test_hits_at_10

def parse_args():
    parser = argparse.ArgumentParser(description='Knowledge Graph Completion with TransE')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                      help='Device to run the model on (cuda or cpu)')
    parser.add_argument('--llm_model', type=str, default='llama3.1:8b',
                      help='LLM model to use')
    parser.add_argument('--n_steps', type=int, default=10,
                      help='Number of steps to run the evolutionary algorithm')
    parser.add_argument('--pop_size', type=int, default=5,
                      help='Population size for the evolutionary algorithm')
    parser.add_argument('--mutation_rate', type=float, default=0.5,
                      help='Mutation rate for the evolutionary algorithm')
    parser.add_argument('--pretrain', action='store_true', default=False,
                      help='Whether to use pretrain model')
    parser.add_argument('--batch_size', type=int, default=512,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                      help='Learning rate')
    parser.add_argument('--embedding_dim', type=int, default=100,
                      help='Dimension of embeddings')
    parser.add_argument('--margin', type=float, default=0.5,
                      help='Margin for loss function')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    return parser.parse_args()

def main(args):
    # Set seed
    set_seed(args.seed)

    save_dir = f"checkpoints/{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    kg_train, kg_val, kg_test = load_wn18rr("./data/WN18RR")
    print(f"Dataset loaded: {kg_train.head_idx.shape} training triples, {kg_val.head_idx.shape} validation triples, {kg_test.head_idx.shape} test triples, {kg_train.n_ent} entities, {kg_train.n_rel} relations")

    # Initialize model
    embedding_model = TransEModel(
        emb_dim=args.embedding_dim,
        n_entities=kg_train.n_ent,
        n_relations=kg_train.n_rel,
        dissimilarity_type='L2'
    ).to(device)

    # Initialize training components
    criterion = MarginLoss(margin=args.margin)
    optimizer = torch.optim.Adam(embedding_model.parameters(), lr=args.learning_rate)
    sampler = BernoulliNegativeSampler(kg_train)
    dataloader = DataLoader(kg_train, batch_size=args.batch_size, use_cuda='all' if device.type == 'cuda' else None)

    # Train model
    if device.type == 'cuda':
        cuda.empty_cache()
        embedding_model.cuda()
        criterion.cuda()

    # Train and evaluate
    print(args.pretrain)
    if  args.pretrain:
        if os.path.exists(f"{save_dir}/transE_model_checkpoint.pth"):
            embedding_model.load_state_dict(torch.load(f"{save_dir}/transE_model_checkpoint.pth")["state_dict"])
        else:
            trainer(embedding_model, dataloader, criterion, optimizer, sampler, num_epochs=args.epochs, device=device)
    evaluate_on_test_set(embedding_model, kg_test)

    n_steps = args.n_steps
    pop_size = args.pop_size
    mutation_rate = args.mutation_rate
    llm_model = args.llm_model

    ollama.pull(llm_model)
    population, fitness_scores = evolutionary_algorithm(kg_train, embedding_model, llm_model=llm_model, batch_size=args.batch_size, optimizer=optimizer, criterion=criterion, 
                                                        generations=n_steps, pop_size=pop_size, mutation_rate=mutation_rate, device=device)
    best_candidate = population[np.argmin(fitness_scores)]
    torch.save({"population": population, "fitness_scores": fitness_scores}, f"{save_dir}/last_population.pth.tar")
    df = pd.DataFrame(best_candidate, columns=['from', 'rel', 'to'])
    kg_candidate = KnowledgeGraph(kg={'heads': torch.concat([kg_train.head_idx, torch.tensor(df['from']).long()]), 
                                          'tails': torch.concat([kg_train.tail_idx, torch.tensor(df['to']).long()]), 
                                          'relations': torch.concat([kg_train.relations, torch.tensor(df['rel']).long()])}, 
                                      ent2ix=kg_train.ent2ix, rel2ix=kg_train.rel2ix, 
                                      dict_of_heads=kg_train.dict_of_heads, 
                                      dict_of_tails=kg_train.dict_of_tails, 
                                      dict_of_rels=kg_train.dict_of_rels)
    best_dataloader = DataLoader(kg_candidate, batch_size=args.batch_size)
    sampler = BernoulliNegativeSampler(kg_candidate)
    trainer(embedding_model, best_dataloader, criterion, optimizer, sampler, num_epochs=args.epochs, device=device)
    evaluate_on_test_set(embedding_model, kg_test)

    torch.save({"state_dict": embedding_model.state_dict()}, f"{save_dir}/transE_model_post_EA_checkpoint.pth")


if __name__ == "__main__":
    args = parse_args()
    main(args)
