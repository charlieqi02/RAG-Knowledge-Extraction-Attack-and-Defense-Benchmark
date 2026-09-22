# Benchmarking Knowledge-Extraction Attacks and Defenses on Retrieval-Augmented Generation (KDD 2026)

![Overview](paper.png)

This repository contains the code for the paper *"Benchmarking Knowledge-Extraction Attacks and Defenses on Retrieval-Augmented Generation"* — **accepted at the KDD 2026 Benchmark and Dataset track**. It provides a modular pipeline for evaluating adversarial knowledge-extraction attacks and defenses on RAG systems across multiple datasets, LLM generators, and embedding models.

**Paper:** [arXiv:2602.09319](https://arxiv.org/pdf/2602.09319) ([KDD 2026 proceedings](https://doi.org/10.1145/3770855.3817524))

## Table of Contents

- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
- [Configuration](#configuration)
- [Running Experiments](#running-experiments)
- [Supported Attacks](#supported-attacks)
- [Supported Defenses](#supported-defenses)
- [Datasets](#datasets)
- [Evaluation](#evaluation)
- [Citation](#citation)

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
    ├── attacks/                #   Per-dataset, per-attack base scripts (building blocks)
    │   ├── enron/              #     {randemb,randtoken,randtext,dgea,copybreak,ikea,utility}.bash
    │   ├── harrypotter/        #     (same set)
    │   ├── healthcare/         #     (same set)
    │   └── pokemon/            #     (same set)
    │
    ├── main-table/             #   Main results: 4 datasets x 6 attacks x 4 defenses
    │   └── run.sh              #     (None / Threshold / Summary / SystemBlock)
    ├── queryblock/             #   QueryBlock defense across all attacks/datasets
    │   └── run.sh
    ├── efficiency/             #   Per-query wall time + API cost benchmark
    │   └── run.sh
    ├── query-diversity/        #   Threshold-defense sweep for embedding-based attacks
    │   └── run.sh
    │
    ├── ablation-command/       #   Ablation: instruction-prompt x generator
    │   └── run.sh
    ├── ablation-generator/     #   Ablation: closed + open RAG generators
    │   └── run.sh
    ├── ablation-emb/           #   Ablation: retriever + attacker embedding model
    │   └── run.sh              #     (white-box MiniLM, white-box BGE-large, black-box)
    ├── ablation-chunking/      #   Ablation: chunk size / strategy variants
    │   ├── run.sh
    │   └── chunked/            #     {enron,healthcare,harrypotter}-chunk/*.bash
    │
    ├── target-extraction/      #   Private-information extraction targets per dataset
    │   └── {enron,harrypotter,healthcare,pokemon}.sh
    │
    ├── rebuttals/              #   Additional experiments for paper rebuttals
    │   ├── multi-seeds/        #     Variance over multiple random seeds
    │   ├── defense-cost/       #     Cost-of-defense for a fixed attack
    │   ├── defense-utility/    #     Utility under each defense (benign queries)
    │   ├── defense-combine/    #     Combined defenses (QBRetrSum, RetrSumSys)
    │   ├── sensitive-audit/    #     Sensitive-attribute audit per dataset
    │   ├── multilingual/       #     Chinese + Vietnamese medical corpora
    │   └── new-defenses/       #     SAGE, VAGUE-GATE
    │
    ├── get_emb_stats.sh        #   Embedding distribution stats on WikiText
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

All experiment scripts live in `./scripts/`. The layout follows a two-tier pattern:

- **`scripts/attacks/{dataset}/{attack}.bash`** — atomic, per-dataset per-attack invocations of `pipeline.py`. Use these as building blocks; they accept extra args via `"$@"`.
- **`scripts/<experiment>/run.sh`** — orchestrators that sweep the building blocks across datasets, attacks, generators, defenses, etc.

Any flag passed to a `run.sh` is forwarded to every underlying attack script, e.g.
`bash scripts/main-table/run.sh --rg_generator llama3-8B-I`.

#### 1. Single Attack on a Single Dataset

```bash
# DGEA on Enron, no defense
bash scripts/attacks/enron/dgea.bash

# Same, but override generator and instruction prompt
bash scripts/attacks/enron/dgea.bash \
    --rg_generator "gpt4o-mini" \
    --ak_command_prompt "attack_templates/jailbreak.txt"
```

#### 2. Main Results Table

`scripts/main-table/run.sh` — 4 datasets x 6 attacks x 4 defenses (`None`, `Threshold`, `Summary`, `SystemBlock`). `QueryBlock` is run separately.

```bash
bash scripts/main-table/run.sh
```

#### 3. QueryBlock Defense

`scripts/queryblock/run.sh` — runs all 6 attacks plus the Utility baseline against QueryBlock on every dataset.

```bash
bash scripts/queryblock/run.sh
```

#### 4. Efficiency Benchmarking

`scripts/efficiency/run.sh` — per-query wall time and API cost across all datasets, using the OpenAI-direct generator endpoint for accurate token accounting.

```bash
bash scripts/efficiency/run.sh
bash scripts/efficiency/run.sh --defense None
```

#### 5. Query Diversity (Threshold Sweep)

`scripts/query-diversity/run.sh` — measures how the Threshold defense affects DGEA and RandEmb across `--df_threshold ∈ {0.3, 0.5, 0.7}` for all 4 datasets.

```bash
bash scripts/query-diversity/run.sh
```

#### 6. Ablation Studies

| Script | What it varies |
|--------|----------------|
| `scripts/ablation-command/run.sh` | Generator x instruction-prompt template x dataset x attack |
| `scripts/ablation-generator/run.sh` | RAG generators (closed + open: GPT-4o/-mini, Llama3-8B-I, Qwen2.5-7B, Claude Sonnet) |
| `scripts/ablation-emb/run.sh` | Retriever and attacker embedding model (white-box MiniLM, white-box BGE-large, black-box BGE/GTE) |
| `scripts/ablation-chunking/run.sh` | Chunk size / strategy via the `chunked/` dataset variants |

```bash
bash scripts/ablation-command/run.sh
bash scripts/ablation-generator/run.sh
bash scripts/ablation-emb/run.sh
bash scripts/ablation-chunking/run.sh
```

#### 7. Target Extraction

Per-dataset private-information extraction targets:

```bash
bash scripts/target-extraction/enron.sh
bash scripts/target-extraction/healthcare.sh
bash scripts/target-extraction/harrypotter.sh
bash scripts/target-extraction/pokemon.sh
```

#### 8. Rebuttal Experiments

Each subfolder under `scripts/rebuttals/` has its own `run.sh`:

| Subfolder | Purpose |
|-----------|---------|
| `multi-seeds/` | Variance across seeds for (defense x attack) combos on HarryPotter + Pokemon |
| `defense-cost/` | Cost of each defense on a fixed attack (RandomToken on HarryPotter) |
| `defense-utility/` | RAG utility on benign queries under each defense |
| `defense-combine/` | Combined defenses: `QBRetrSum`, `RetrSumSys` |
| `sensitive-audit/` | Sensitive-attribute audit per dataset |
| `multilingual/` | Extraction on Chinese + Vietnamese medical corpora |
| `new-defenses/sage/` | SAGE defense (Sensitive-Attribute Gating) |
| `new-defenses/vague-gate/` | VAGUE-GATE defense |

```bash
bash scripts/rebuttals/multi-seeds/run.sh
bash scripts/rebuttals/new-defenses/sage/run.sh
```

#### 9. Utilities

```bash
bash scripts/get_emb_stats.sh   # Embedding-distribution stats on WikiText
bash scripts/reduce.sh          # t-SNE dimensionality reduction over logged queries
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

## Citation

If you use this benchmark or the code in your research, please cite our [KDD paper](https://doi.org/10.1145/3770855.3817524) ([arXiv mirror](https://arxiv.org/abs/2602.09319)):

```bibtex
@inproceedings{10.1145/3770855.3817524,
    author = {Qi, Zhisheng and Sahu, Utkarsh and Ma, Li and Han, Haoyu and Rossi, Ryan and Dernoncourt, Franck and Halappanavar, Mahantesh and Ahmed, Nesreen and Dong, Yushun and Zhao, Yue and Zhang, Yu and Wang, Yu},
    title = {Benchmarking Knowledge-Extraction Attack and Defense on Retrieval-Augmented Generation (RAG)},
    year = {2026},
    isbn = {9798400722592},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3770855.3817524},
    doi = {10.1145/3770855.3817524},
    abstract = {Retrieval-Augmented Generation (RAG) has become a cornerstone of knowledge-intensive applications, including enterprise chatbots, healthcare assistants, and agentic memory systems. However, recent studies show that knowledge-extraction attacks can recover sensitive knowledge-base content through maliciously crafted queries, raising serious privacy and intellectual-property concerns. While prior work has explored individual attack and defense techniques, the research landscape remains fragmented across retrievers, generators, and evaluations based on non-standardized metrics and datasets. To address this gap, we introduce the first systematic benchmark for knowledge-extraction attacks on RAG systems. Our benchmark covers broad attack/defense strategies, representative retrieval embedding models, open/closed-source generators, (non) graph-based indexing, all evaluated under a unified framework with standardized protocols across multiple datasets spanning diverse languages. By consolidating the experimental landscape and enabling reproducible, comparable evaluation, this benchmark provides actionable insights and a practical foundation for developing privacy-preserving RAG systems in the face of emerging knowledge extraction threats.},
    booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2},
    pages = {9718–9729},
    numpages = {12},
    keywords = {retrieval-augmented generation, knowledge-extraction attack},
    location = {Republic of Korea},
    series = {KDD '26}
}}
```
