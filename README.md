# Large Language Models as Financial Analysts: Sector-Aware Reasoning

Official implementation of the **sector-aware LLM framework for asset selection**, as described in:

> Kim, H., Jeong, J., Ko, H. et al. *Large Language Models as Financial Analysts: Sector-Aware Reasoning for Investment Decisions.* **Computational Economics** (2026).
> DOI: [10.1007/s10614-026-11329-4](https://doi.org/10.1007/s10614-026-11329-4)

## Overview

We frame next-month return prediction as **binary classification** with probabilistic output:

- Given firm characteristics and GICS sector information, the LLM outputs a **Return Movement Score** `p̂ ∈ [0, 1]` (≥ 0.5 = expected increase).

**Key idea: Sector-aware prompting**

- The LLM assumes a **sector-specific analyst role** based on the asset's GICS sector.
- Prompts follow a structured template requesting both the score and a brief risk-return rationale.

**Tie-breaking**: When assets share identical scores, we use **perplexity** as a confidence measure (lower = more confident).

## Pipeline

1. **LLM Inference** (`vllm/run.py`)
   - Reads monthly question table, runs vLLM inference (default: Llama 3 8B Instruct)
   - Outputs JSONL with scores, rationales, and token-level statistics
   - Output: `vllm/outputs/result*_run*.jsonl`

2. **Score Extraction** (`vllm/json2csv.py`)
   - Parses JSONL to extract scores and rationales
   - Merges with features by `(permno, year, month)`
   - Output: `vllm/outputs/parsed_*.csv`

3. **Portfolio Backtests** (`portfolio/main.py`)
   - Long-only portfolios (equal-weight / value-weight)
   - Mean-variance optimization (max Sharpe / min risk / max return)
   - Output: `portfolio/outputs/*.csv`

## Environment

```bash
conda env create -f environment.yml
conda activate sector
```

- `environment.yml`: Minimal dependencies (recommended)
- `environment.lock.yml`: Full export with exact versions for reproducibility

## Data

- `data/llm_questions_2012_2021.csv` - Monthly prompts for LLM input
- `data/question_features_2011_2021.csv` - Features table for merging (key: `permno, year, month`)
- `data/prices_2012_2021.csv` - Price data for backtests
- `data/permno_monthly_meta.csv` - Monthly metadata (`prc`, `ret`, `shrout`, etc.)

See [data/README.md](data/README.md) for details.

## Usage

```bash
# 1. LLM inference
python vllm/run.py --data-dir data --output-dir vllm/outputs

# 2. Parse results
python vllm/json2csv.py --input-dir vllm/outputs --question-csv data/question_features_2011_2021.csv

# 3. Run backtests
python portfolio/main.py
```

## Prompt Variants

By default, `vllm/run.py` runs two variants:

- **With sector**: Sector-specific analyst role using GICS
- **Without sector**: Generic analyst role (ablation)

Use `--csv-base` and `--csv-except` to specify different input tables for each variant.

## Citation

```bibtex
@article{kim2026llm_sector_aware,
  title   = {Large Language Models as Financial Analysts: Sector-Aware Reasoning for Investment Decisions},
  author  = {Kim, Hyeonjin and Jeong, Jiwoo and Ko, Hyungjin and Lee, Woojin},
  journal = {Computational Economics},
  year    = {2026},
  doi     = {10.1007/s10614-026-11329-4}
}
```
