# 🚀 FLAT-LLM

This is the official PyTorch implementation of **FLAT-LLM** **F**ine-grained **L**ow-rank **A**ctivation Space **T**ransformation for Large Language Model Compression [arxiv](https://arxiv.org/pdf/2505.23966)

---

## 📦 Environment Setup

Installation instructions can be found in [INSTALL.md](INSTALL.md).

---

## 🛠️ Run the Code

All scripts for reproducing our main results (Table 1) are available in the [`scripts`](scripts) directory.

### 🔍 Importance-Preserving Rank Selection (IPRS)

1. Run `llama_bi.sh` to compute decoder-wise importance scores.
2. Run `compute_rank.py` to 
    - Allocate ranks with our IPRS algorithm according to the importance scores.
    - Compute the compression ratio for V,O,MLP layers (Q,K are not pruned) according to required total compresstion ratio.

### ✂️ FLAT-LLM Pruning

Run one of the following scripts to prune and evaluate the corresponding model:
- `llama_7b.sh` # use 1 A100 40GB
- `llama_13b.sh` # use 1 A100 40GB
- `llama_70b.sh` # use 4 A100 40GB
- `mistral.sh` # use 1 A100 40GB

These reproduce the perplexity results reported in Table 1 of the paper when using wikitext2 for calibration.

---

## 🔧 Command-Line Arguments

### 📦 Model and Dataset
- `--model`: Name or path of the LLM to prune. Choices: `meta-llama/Llama-2-7b-hf`, `meta-llama/Llama-2-13b-hf`, `meta-llama/Llama-2-70b-hf`, `mistralai/Mistral-7B-v0.1`
- `--dataset`: Calibration dataset. Choices: `wikitext2`, `c4`, `alpaca`.
- `--cache_dir`: Directory to cache model weights.

### ⚙️ Pruning Configuration
- `--prune_method`: Pruning stage. Options:
  - `bi`: Rank allocation via importance scores.
  - `flatllm`: Final pruning using head-wise PCA.
- `--sparsity_ratio`: Target sparsity level (as an integer percentage).
- `--tol`: Tolerance threshold on cumulative eigenvalues. Default: `0.96`. (this hyper-para is only for monitoring the calibration, not used in the algorithm)
- `--bi_score`: Path to save/load the importance scores/allocated ranks.
- `--seed`: Random seed for reproducibility.
- `--nsamples`: Number of calibration samples.
- `--save`: Path to save logs.
- `--save_model`: Path to save the pruned model.

---

## 📊 Evaluation

### 🧠 Zero-Shot Evaluation

We evaluate zero-shot downstream task performance using the [EleutherAI LM Harness](https://github.com/EleutherAI/lm-evaluation-harness). Please use the modified code for zero-shot/few-shot evaluation in [lm_eval](https://github.com/TTTTTTris/lm_eval) repo.

### ⚡ Inference Speedup

To benchmark inference speedup, we build upon the evaluation framework from [SliceGPT](https://github.com/microsoft/TransformerCompression).

---

## 📄 License

This project is licensed under the MIT License. 
