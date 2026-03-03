<p align="center">
  <h2 align="center"><strong>Large Language Models as Financial Analysts: Sector-Aware Reasoning</strong></h2>
  <h3 align="center"><strong>Computational Economics (2026)</strong></h3>
</p>

<p align="center">
  <a href="https://rdcu.be/e4Ynd">📄 Paper</a> &nbsp;|&nbsp;
  <a href="https://link.springer.com/article/10.1007/s10614-026-11329-4">🔗 Springer</a>
</p>

<!-- Optional: add a figure under CE/SectorLLM/assets/overview.png -->
<!--
<p align="center">
  <img src="assets/overview.png" width="85%">
</p>
-->

<br/>

This repository provides the **official implementation of a sector-aware LLM framework for asset selection**, introduced in:

> Kim, H., Jeong, J., Ko, H. et al. *Large Language Models as Financial Analysts: Sector-Aware Reasoning for Investment Decisions.* **Computational Economics** (2026). DOI: 10.1007/s10614-026-11329-4

## 📢 Updates

- **2026**: Code released.

## 🔍 Framework Overview

We frame next-month return prediction as **binary classification** with a probabilistic output. Given firm characteristics and GICS sector information, the LLM outputs a **Return Movement Score** `p̂ ∈ [0, 1]` (≥ 0.5 = expected increase).

**Key idea: sector-aware prompting**

- The LLM assumes a **sector-specific analyst role** based on the asset's GICS sector.
- Prompts follow a structured template requesting both the score and a brief risk-return rationale.

**Tie-breaking**: When assets share identical scores, we use **perplexity** as a confidence measure (lower = more confident).

## Getting Started

### 🛠️ Environment Setup

```bash
git clone https://github.com/your-repo/SectorLLM.git
cd SectorLLM

conda env create -f environment.yml
conda activate sector
```

- `environment.yml`: Minimal dependencies (recommended)
- `environment.lock.yml`: Full export with exact versions for reproducibility

### 📁 Data Preparation

Place the following files in `data/`:

- `llm_questions_2012_2021.csv` - Monthly prompts for LLM input
- `question_features_2011_2021.csv` - Features table for merging (key: `permno, year, month`)
- `prices_2012_2021.csv` - Price data for backtests
- `permno_monthly_meta.csv` - Monthly metadata (`prc`, `ret`, `shrout`, etc.)

See [data/README.md](data/README.md) for details.

### 🤖 LLM Inference

Run sector-aware prompting with vLLM (default: Llama 3 8B Instruct).

```bash
python vllm/run.py --data-dir data --output-dir vllm/outputs
```

By default, this runs two prompt variants:
- **With sector**: Sector-specific analyst role using GICS
- **Without sector**: Generic analyst role (ablation)

### 📊 Score Extraction

Parse JSONL outputs and merge with features.

```bash
python vllm/json2csv.py --input-dir vllm/outputs --question-csv data/question_features_2011_2021.csv
```

### 🚀 Portfolio Backtests

Run backtests with long-only (EW/VW) and mean-variance optimization strategies.

```bash
python portfolio/main.py
```

## 📝 Citation

```bibtex
@article{kim2026llm_sector_aware,
  title   = {Large Language Models as Financial Analysts: Sector-Aware Reasoning for Investment Decisions},
  author  = {Kim, Hyeonjin and Jeong, Jiwoo and Ko, Hyungjin and Lee, Woojin},
  journal = {Computational Economics},
  year    = {2026},
  doi     = {10.1007/s10614-026-11329-4}
}
```
