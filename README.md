# CMNA'26: Stance-Aware Text Retrieval

This repository contains the codebase and reproducibility artifacts for our paper on stance-aware text retrieval. It provides the full pipeline to train, evaluate, and ablate models using Homogeneous, Mixed, and Mixed+Augmented curricula.

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/as9122/cmna26-stance-aware-text-retrieval.git
   cd cmna26-stance-aware-text-retrieval
   ```

2. **Install required dependencies:**
We recommend using a virtual environment (Python 3.10+).
    ```bash
    pip install -r requirements.txt

    ```
##  Pipeline Execution

The codebase is modular. You can run the entire pipeline from scratch, or simply use our pre-trained Hugging Face adapters to instantly reproduce the evaluation metrics.

### Fine-Tuning (`src/tune.py`)

Trains new LoRA adapters on the training corpora using the specified data-centric curricula and a custom batch sampler.

```bash
# Example: Train Qwen3-8B using the Mixed curriculum with LoRA rank 16
python src/tune.py --model_name Qwen3-Embedding-8B --strategy Mixed --rank 16

```

### Precomputing Raw Embeddings (`src/precompute_embs.py`)

Generates and saves static corpus embeddings for either the base models or the fine-tuned adapters.

```bash
# Example: Precompute embeddings for the AVeriTeC dataset
python src/precompute_embs.py \
    --base_model Qwen3-Embedding-8B \
    --dataset averitec \
    --adapter as9122/qwen3-8b-embedding-stance-mixed

```

### Dense Retrieval & Reranking (`src/retrieval_with_reranker.py`)

Evaluates models across the target datasets by performing dense retrieval followed by Cross-Encoder reranking.

```bash
# Example: Evaluate a local adapter across all datasets
python src/retrieval_with_reranker.py \
    --base_model Qwen3-Embedding-8B \
    --adapter ./results/finetuned_models/Qwen3-Embedding-8B_Mixed_final

# Example: Evaluate an adapter hosted directly on Hugging Face
python src/retrieval_with_reranker.py \
    --base_model Qwen3-Embedding-8B \
    --adapter as9122/qwen3-8b-embedding-stance-mixed

```

### Retrieval Metrics (`src/retrieval_metrics.py`)

Processes the raw retrieval embeddings to calculate standard information retrieval metrics such as NDCG and Precision.

```bash
# Automatically scans the results directory and aggregates the retrieval metrics
python src/retrieval_metrics.py
```

### Word Ablation (`src/ablation.py`)

Systematically ablates query and instruction words to generate raw similarity deltas for model sensitivity testing.

```bash
# Example: Run the ablation engine on the AVeriTeC dataset using a Hugging Face adapter
python src/ablation.py \
    --base_model Qwen3-Embedding-8B \
    --dataset averitec \
    --adapter as9122/qwen3-8b-embedding-stance-mixed

```

### Ablation Metrics (`src/ablation_metrics.py`)

Aggregates the raw ablation data to compute the final Relative Instruction Sensitivity (RIS), Relative Claim Sensitivity (RCS), and Directional Impact (DI) metrics.

```bash
# Automatically scans the results directory and outputs ablation_summary_stats.csv
python src/ablation_metrics.py

```

## Repository Structure

* `src/`: Core pipeline scripts.
    * `tune.py`: LoRA fine-tuning with custom batching.
    * `precompute_embs.py`: Static corpus embedding generation.
    * `retrieval_with_reranker.py`: Dense retrieval + CrossEncoder evaluation.
    * `retrieval_metrics.py`: IR metric calculation.
    * `ablation.py`: Word-level ablation engine.
    * `ablation_metrics.py`: Calculation of RIS, RCS, and DI.


* `utils/`: Supplemental and visualisation tools.
    * `ablation_figs.py` / `retrieval_figs.py`: Code for generating plots relating to ablation/retrieval respectively.
    * `ablation_heatmaps.py`: Code for generating ablation heatmaps.
    * `ablation_tables.py` / `retrieval_tables.py`: Code for formatting metrics into CEURART LaTeX tables.