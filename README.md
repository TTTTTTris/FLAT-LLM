# 🚀 FLAT-LLM

This is the official PyTorch implementation of **FLAT-LLM** (**F**ine-grained **L**ow-rank **A**ctivation Space **T**ransformation for Large Language Model Compression)

---

## 📦 Environment Setup

Installation instructions can be found in [INSTALL.md](INSTALL.md).

---

## 🛠️ Run the Code

All scripts for reproducing our main results (Table 1) are available in the [`scripts`](scripts) directory.

### 🔍 Importance-Preserving Rank Selection (IPRS)

Run `llama_bi.sh` to compute decoder-wise importance scores and allocate ranks using the IPRS algorithm.

### ✂️ FLAT-LLM Pruning

Run one of the following scripts to prune and evaluate the corresponding model:
- `llama_7b.sh`
- `llama_13b.sh`
- `llama_70b.sh`
- `mistral.sh`

These reproduce the perplexity results reported in Table 1 of the paper.

---

## 🔧 Command-Line Arguments

### 📁 Model and Dataset
- `--model`: Name or path of the LLM to prune.
- `--dataset`: Calibration dataset. Choices: `wikitext2`, `c4`, `alpaca`.
- `--cache_dir`: Directory to cache model weights.

### ✂️ Pruning Configuration
- `--prune_method`: Pruning stage. Options:
  - `bi`: Rank allocation via importance scores.
  - `flatllm`: Final pruning using head-wise PCA.
- `--sparsity_ratio`: Target sparsity level (as an integer percentage).
- `--tol`: Tolerance threshold on cumulative eigenvalues. Default: `0.96`.
- `--bi_score`: Path to the file containing precomputed importance scores.

### 🧪 Calibration Settings
- `--seed`: Random seed for reproducibility.
- `--nsamples`: Number of calibration samples. Default: `128`.

### 💾 Evaluation and Saving
- `--eval_zero_shot`: Run zero-shot evaluation on downstream tasks.
- `--save`: Path to save evaluation results.
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
