# EvokGE: Evolutionary Knowledge Graph Embedding with LLM Guidance

EvokGE is a novel approach that combines evolutionary algorithms with Large Language Models (LLMs) to enhance knowledge graph embedding and link prediction. The project implements an evolutionary algorithm that uses LLM-guided mutations to optimize knowledge graph embeddings, specifically focusing on the WN18RR dataset.

## Features

- Evolutionary algorithm for knowledge graph embedding optimization
- LLM-guided mutation operator for generating diverse and semantically consistent triples
- Integration with TransE model for knowledge graph embedding
- Support for link prediction evaluation
- GPU acceleration support

## Requirements

- Python 3.x
- PyTorch
- torchkge
- transformers
- ollama
- pandas
- numpy
- tqdm

## Installation

1. Clone the repository:
```bash
git clone https://github.com/rkhosrowshahli/evokge.git
cd evokge
```

2. Install the required dependencies:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

3. Download the WN18RR dataset:
```bash
mkdir -p data/WN18RR
# Download WN18RR dataset to data/WN18RR directory
```

## Usage

The main script can be run with the following command:

```bash
python main.py --llm_model <model_name> --batch_size <batch_size> --generations <num_generations> --pop_size <population_size> --mutation_rate <mutation_rate>
```

### Parameters

- `--llm_model`: Name of the LLM model to use for mutation (default: "llama2 or llama3.1:8b")
- `--batch_size`: Batch size for training (default: 32)
- `--generations`: Number of evolutionary generations (default: 10)
- `--pop_size`: Population size for the evolutionary algorithm (default: 100)
- `--mutation_rate`: Mutation rate for the evolutionary algorithm (default: 0.5)
- `--device`: Device to run the model on (default: "cuda" if available, else "cpu")

### Example

Here's an example of running the code with specific parameters:

```bash
# Run with default parameters
python main.py

# Run with custom parameters
python main.py --llm_model llama3.1:8b \
               --batch_size 64 \
               --n_steps 10 \
               --pop_size 10 \
               --mutation_rate 0.5 \
               --device cuda
```

This example:
- Uses the llama2 or llama3.1:8b model for mutation guidance
- Sets batch size to 64 for training
- Runs for 10 steps
- Uses a population size of 100
- Sets mutation rate to 0.5 (50% chance of mutation)
- Explicitly uses CUDA for GPU acceleration

## How It Works

1. **Initialization**: The algorithm starts by loading the WN18RR dataset and initializing a TransE model.
2. **Candidate Generation**: Generates an initial population of candidate knowledge graph triples.
3. **Fitness Evaluation**: Evaluates each candidate using the TransE model and link prediction metrics.
4. **Evolutionary Process**:
   - Selection: Selects the best candidates based on fitness scores
   - Crossover: Combines selected candidates to create offspring
   - Mutation: Uses LLM guidance to mutate the offspring while maintaining semantic consistency
5. **Evaluation**: The final population is evaluated on the test set to measure performance.

## Results

The algorithm outputs the following metrics:
- Mean Reciprocal Rank (MRR)
- Hits@1, Hits@3, Hits@10
- Mean Rank

## License

This project is licensed under the MIT License - see the LICENSE file for details.