# Benchmarking Knowledge-Extraction Attacks and Defenses on Retrieval-Augmented Generation

This repository contains the code for the paper *"Benchmarking Knowledge-Extraction Attacks and Defenses on Retrieval-Augmented Generation"*. It provides a modular pipeline for evaluating adversarial knowledge-extraction attacks and defenses on RAG systems across multiple datasets, LLM generators, and embedding models.

## Table of Contents

- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
- [Configuration](#configuration)
- [Running Experiments](#running-experiments)
- [Supported Attacks](#supported-attacks)
- [Supported Defenses](#supported-defenses)
- [Datasets](#datasets)
- [Evaluation](#evaluation)

## Project Structure

```
Extraction-AD-Pipeline/
├── pipeline.py                 # Main entry point for all experiments
├── set_env.sh                  # Environment variable setup script
├── keys.yaml                   # API keys configuration (not tracked)
│
├── args/                       # Command-line argument definitions
│   ├── pipeline_args.py        #   Pipeline-level args (dataset, rag, attack, defense, seed, gpu)
│   ├── rag_args.py             #   RAG system args (retriever, generator, top-k, temperature)
│   ├── dataset_args.py         #   Dataset-specific args
│   ├── attack_args.py          #   Attack-specific args (query budget, models, templates)
│   └── defense_args.py         #   Defense-specific args (thresholds, prompts)
│
├── attacks/                    # Attack method implementations
│   ├── base.py                 #   Abstract base class for all attacks
│   ├── dgea.py                 #   DGEA: Dynamic Greedy Embedding Attack
│   ├── copybreak.py            #   CopyBreak: Feedback-guided agent attack
│   ├── ikea.py                 #   IKEA: Implicit Knowledge Extraction Attack
│   ├── random.py               #   Random baselines (RandomEmb, RandomToken, RandomText)
│   └── utility.py              #   Utility query baseline (benign queries)
│
├── defenses/                   # Defense method implementations
│   ├── base.py                 #   Defense base class (also handles None, Summary, Threshold, SystemBlock)
│   └── queryblock.py           #   QueryBlock: LLM-based malicious query detection
│
├── rags/                       # RAG system implementations
│   ├── base.py                 #   Abstract RAG base class
│   └── txt_rag.py              #   TextRAG: Text-based RAG with ChromaDB vector store
│
├── kedatasets/                  # Dataset loading and processing
│   ├── ke_dataset.py           #   Main dataset class
│   ├── _data_load.py           #   Per-dataset loading functions
│   └── _utils.py               #   Text splitting and indexing utilities
│
├── tools/                      # Shared utilities
│   ├── get_llm.py              #   LLM model loader (OpenAI, Azure, Llama, GCP)
│   ├── get_embedding.py        #   Embedding model loader
│   ├── _llm_engines.py         #   LLM engine implementations
│   ├── _llama_engines.py       #   Local Llama model engine
│   ├── _embedding_models.py    #   Embedding model implementations
│   ├── attacks.py              #   Attack utility functions (similarity, parsing, refusal detection)
│   ├── parse_response.py       #   Response parsing for extraction evaluation
│   ├── train.py                #   Seed setting and save directory management
│   └── args.py                 #   Argument parsing orchestration
│
├── recorder/                   # Experiment recording and evaluation
│   ├── recorder.py             #   Per-query result recording to JSONL
│   ├── evaluator.py            #   Batch evaluation across experiment logs
│   ├── evaluation.py           #   Evaluation metric computation
│   ├── tsne_vis.py             #   t-SNE visualization of query embeddings
│   └── tsne_reduce.py          #   Dimensionality reduction for visualization
│
├── prompts/                    # Prompt templates
│   ├── textrag/                #   RAG system prompts (system.txt, template.txt)
│   ├── attack_templates/       #   Attack instruction variants (simple, median, jailbreak)
│   ├── defense/                #   Defense prompts (query_block_system.txt, summary_*.txt)
│   ├── dgea/                   #   DGEA-specific prompts
│   ├── copybreak/              #   CopyBreak-specific prompts (explore, exploit templates)
│   ├── ikea/                   #   IKEA-specific prompts (anchor generation, mutation)
│   └── random/                 #   Random attack generation prompts
│
├── data/                       # Datasets and vector databases
│   ├── Enron/                  #   Enron email corpus (~500k documents)
│   ├── HarryPotter/            #   Harry Potter text (~26k chunks)
│   ├── HealthCareMagic/        #   Medical Q&A (~100k records)
│   ├── Pokemon/                #   Pokemon dataset (~1k entries)
│   ├── Sampled/                #   Chunked/sampled dataset variants
│   └── databases/              #   Persisted ChromaDB vector stores
│
├── extra_data/                 # Auxiliary data (e.g., WikiText samples)
├── logs/                       # Experiment output logs (auto-created)
│
└── scripts/                    # Bash scripts for running experiments
    ├── ablation-command/       #   Ablation study: attacks x instruction prompts x generators
    ├── query-diversity/        #   Query diversity and threshold experiments
    ├── queryblock/             #   QueryBlock defense experiments
    ├── efficiency/             #   Efficiency benchmarking experiments
    ├── target_extraction/      #   Private information extraction scripts
    └── reduce.sh               #   t-SNE dimensionality reduction
```

## Environment Setup

### 1. Create a Conda Environment

```bash
conda create -n ke-rag python=3.10 -y
conda activate ke-rag
```

### 2. Install Dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # adjust CUDA version as needed

pip install \
    openai \
    anthropic[vertex] \
    langchain \
    langchain-community \
    chromadb \
    transformers \
    sentence-transformers \
    rouge \
    rouge-score \
    scikit-learn \
    matplotlib \
    pandas \
    numpy \
    tqdm \
    pyyaml \
    pydantic \
    smolagents
```

### 3. Configure API Keys

Create a `keys.yaml` file in the project root with your API credentials:

```yaml
llm:
  azure:
    gpt4o-mini:
      model_name: <your-azure-model-deployment-name>
      api_key: <your-azure-api-key>
      api_version: "2024-09-01-preview"
      azure_endpoint: <your-azure-endpoint>
    gpt4o:
      model_name: <your-azure-model-deployment-name>
      api_key: <your-azure-api-key>
      api_version: "2024-09-01-preview"
      azure_endpoint: <your-azure-endpoint>
  openai:
    gpt4o-mini:
      model_name: gpt-4o-mini
      api_key: <your-openai-api-key>
      api_version: "2024-09-01-preview"
  gcp:
    claude-4-5-sonnet:
      model_name: <model-name>
      project_id: <your-gcp-project-id>
      region: <your-gcp-region>

embedding:
  azure:
    text-embedding-3-small:
      model_name: text-embedding-3-small
      api_key: <your-azure-embedding-api-key>
      api_version: "2024-09-01-preview"
      azure_endpoint: <your-azure-embedding-endpoint>
```

### 4. Set Up Environment Variables

Before running any experiment, source the environment setup script from the project root:

```bash
source set_env.sh
```

This sets `PYTHONPATH`, `LOG_DIR`, `DATA_PATH`, `DB_PATH`, `KEYS_PATH`, `PROMPT_PATH`, and `EXTRA_PATH`.

## Configuration

All arguments are organized into five groups and passed via the command line:

| Group | Prefix | Key Arguments |
|-------|--------|---------------|
| Pipeline | (none) | `--dataset`, `--rag`, `--attack`, `--defense`, `--seed`, `--gpu` |
| RAG | `--rg_` | `--rg_retriever`, `--rg_generator`, `--rg_retr_kwargs_topk`, `--rg_gen_kwargs_temperature` |
| Dataset | `--ds_` | Dataset-specific parameters |
| Attack | `--ak_` | `--ak_max_query`, `--ak_emb_model`, `--ak_llm_model`, `--ak_iterations` |
| Defense | `--df_` | `--df_threshold`, `--df_query_block_system`, `--df_query_block_template` |

All bash scripts accept additional arguments via `"$@"`, so you can override any parameter at the command line.

## Running Experiments

### Basic Usage

```bash
source set_env.sh

python pipeline.py \
    --des "My experiment" \
    --dataset "HealthCareMagic" \
    --rag "TextRAG" \
    --attack "DGEA" \
    --defense "None" \
    --seed 42 \
    --gpu 0 \
    --rg_retriever "MiniLM" \
    --rg_generator "gpt4o-mini" \
    --rg_retr_kwargs_topk 3 \
    --ak_max_query 200
```

### Using Bash Scripts

All experiment scripts are in `./scripts/`. Each individual `.bash` file contains a fully specified experiment command. Master `.sh` files orchestrate multiple runs.

#### 1. Main Experiments (Ablation over Instruction Prompts)

Located in `scripts/ablation-command/`. Each dataset has a subfolder with per-attack scripts:

```
scripts/ablation-command/
├── ab_com.sh                       # Master script: runs all datasets x attacks x prompts x generators
├── ab_com_enron.sh                 # Enron-only subset
├── ab_com_harry.sh                 # HarryPotter-only subset
├── ab_com_health.sh                # HealthCareMagic-only subset
├── ab_com_pokemon.sh               # Pokemon-only subset
├── enron-500k-none/
│   ├── dgea.bash                   # DGEA attack on Enron
│   ├── copybreak.bash              # CopyBreak attack on Enron
│   ├── randemb.bash                # RandomEmb baseline on Enron
│   ├── randtoken.bash              # RandomToken baseline on Enron
│   └── randtext.bash               # RandomText baseline on Enron
├── harrypotter-26k-none/           # Same structure for HarryPotter
├── health-100k-none/               # Same structure for HealthCareMagic
└── pokemon-1k-none/                # Same structure for Pokemon
```

Run a single attack:

```bash
bash scripts/ablation-command/enron-500k-none/dgea.bash
```

Override the generator and instruction prompt:

```bash
bash scripts/ablation-command/enron-500k-none/dgea.bash \
    --rg_generator "gpt4o-mini" \
    --ak_command_prompt "attack_templates/jailbreak.txt"
```

Run all ablation experiments for a dataset:

```bash
bash scripts/ablation-command/ab_com_enron.sh
```

Run the full ablation study (all datasets, attacks, generators, prompts):

```bash
bash scripts/ablation-command/ab_com.sh
```

#### 2. QueryBlock Defense Experiments

Located in `scripts/queryblock/`. Tests all attacks against the QueryBlock defense:

```bash
# Run all attacks on Enron with QueryBlock defense
bash scripts/queryblock/enron.sh

# Run all attacks on HarryPotter with QueryBlock defense
bash scripts/queryblock/harry.sh

# Run a single attack with QueryBlock
bash scripts/queryblock/enron-500k/dgea.bash
```

#### 3. Efficiency Benchmarking

Located in `scripts/efficiency/`. Measures per-query time and cost:

```bash
# Run efficiency benchmarks across all datasets
bash scripts/efficiency/efficiency.sh

# Run a single efficiency test
bash scripts/efficiency/enron-500k/dgea.bash \
    --defense "None" \
    --rg_generator "gpt4o-mini-openai"
```

#### 4. Query Diversity and Threshold Experiments

Located in `scripts/query-diversity/`. Tests how retrieval similarity thresholds affect attack performance:

```bash
# Run query diversity experiments on Enron (includes Threshold defense)
bash scripts/query-diversity/enron.sh

# Batch threshold experiments
bash scripts/query-diversity/qd_thesh1.sh
bash scripts/query-diversity/qd_thesh2.sh
```

#### 5. Target Extraction

Located in `scripts/target_extraction/`. Extracts private information targets from datasets:

```bash
bash scripts/target_extraction/enron.sh
bash scripts/target_extraction/health.sh
```

#### 6. Visualization

Generate t-SNE plots of query embeddings:

```bash
bash scripts/reduce.sh
```

## Supported Attacks

| Attack | Description | Key Reference |
|--------|-------------|---------------|
| **DGEA** | Dynamic Greedy Embedding Attack. Optimizes adversarial tokens to target specific embedding regions via gradient-based search. | [Anderson et al., 2024](https://arxiv.org/abs/2409.08045) |
| **CopyBreak** | Feedback-guided agent attack. Alternates between exploration (random probing) and exploitation (targeted extraction using discovered chunks). | [Li et al., 2024](https://arxiv.org/abs/2411.14110) |
| **IKEA** | Implicit Knowledge Extraction Attack. Uses anchor concepts with trust-region optimization to systematically extract knowledge. | [Qi et al., 2025](https://arxiv.org/abs/2505.15420) |
| **RandomEmb** | Targets random points in the embedding space. Baseline for DGEA. | - |
| **RandomToken** | Constructs queries from randomly sampled vocabulary tokens. | - |
| **RandomText** | Uses an LLM to generate random text as queries. | - |
| **Utility** | Submits benign utility questions from the dataset. Used to measure RAG utility under defenses. | - |

## Supported Defenses

| Defense | Description |
|---------|-------------|
| **None** | No defense applied (baseline). |
| **Summary** | Modifies the RAG generation prompt to force context summarization instead of verbatim reproduction. |
| **Threshold** | Filters retrieved documents by cosine similarity score; only returns documents above `--df_threshold`. |
| **SystemBlock** | Injects a defensive system prompt instructing the LLM to refuse data-leaking requests. |
| **QueryBlock** | Routes each incoming query through a separate LLM classifier that detects and blocks malicious queries before they reach the RAG system. |

## Datasets

| Dataset | Size | Domain | Description |
|---------|------|--------|-------------|
| **Enron** | ~500k docs | Email | Enron email corpus |
| **HarryPotter** | ~26k chunks | Literary | Harry Potter book text |
| **HealthCareMagic** | ~100k records | Medical | Doctor-patient Q&A pairs |
| **Pokemon** | ~1k entries | Tabular | Pokemon attribute data |

Each dataset includes ~1000 utility questions (`utility_questions.jsonl`) for benign performance evaluation. Chunked/sampled variants are available in `data/Sampled/`.

## Evaluation

After experiments complete, results are saved to `logs/` as JSONL files. Use the evaluator to compute metrics across all runs:

```python
from recorder.evaluator import Evaluator

evaluator = Evaluator(
    log_dirs=["./logs"],
    query_budget=200,
    thresh_ss=0.70,    # sentence-level similarity threshold
    thresh_ls=0.70,    # document-level similarity threshold
    mode="attack"      # "attack" or "utility"
)
evaluator.evaluate()
# Results written to ./logs/results.csv
```

### Metrics

| Metric | Description |
|--------|-------------|
| **ASR** | Attack Success Rate: fraction of queries that successfully extract information |
| **REE** | Retrieval Extraction Efficiency |
| **GEE-ss** | Generation Extraction Efficiency (sentence-level similarity) |
| **GEE-ls** | Generation Extraction Efficiency (document-level similarity) |
| **EE-ss** | Extraction Efficiency (sentence-level similarity) |
| **EE-ls** | Extraction Efficiency (document-level similarity) |

### Supported Models

**LLM Generators:** GPT-4o, GPT-4o-mini (Azure/OpenAI), Llama-3-8B-Instruct, Qwen2.5-7B-Instruct (local), Claude Sonnet (GCP Vertex)

**Embedding Models:** all-MiniLM-L6-v2, GTE-base, BGE-large, Nomic-Embed-v1.5, GTE-small
