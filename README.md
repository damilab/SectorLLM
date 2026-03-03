# Large Language Models as Financial Analysts: Sector-Aware Reasoning

This repository implements the **LLM-based sector-aware asset selection framework** described in:

- Kim, H., Jeong, J., Ko, H. et al. *Large Language Models as Financial Analysts: Sector-Aware Reasoning for Investment Decisions.* **Computational Economics** (2026).
- DOI: `10.1007/s10614-026-11329-4`
- Paper page: `https://link.springer.com/article/10.1007/s10614-026-11329-4`

## Conceptual Overview (Paper Terminology)

We model next-month return direction forecasting as a **binary classification** problem with a continuous probability output:

- For asset *i* at time *t*, given firm characteristics at `t-1` and `t` (feature vectors) and sector information (GICS),
  the LLM produces a **Return Movement Score** `p̂_{i,t+1} ∈ [0, 1]` (≥ 0.5 indicates an expected increase).

The key component is **sector-aware prompting**:

- The LLM is assigned a **sector-specific analyst role** conditioned on the asset’s GICS sector.
- The prompt follows a structured template (cf. the paper’s “LLM Input Template for Asset Return Prediction”),
  requesting both the score and a brief rationale focused on return-risk balance.

**Tie-breaking with model confidence**:

- When multiple assets receive identical scores, we tie-break using **perplexity** (lower perplexity = higher confidence).

## Pipeline Mapping (Repo → Methodology)

1) **Sector-conditioned prompt construction + LLM inference** (`vllm/run.py`)
   - Reads a monthly panel-style question table and runs vLLM inference (default model: Llama 3 8B Instruct).
   - Outputs JSONL with score/rationale and token-level statistics used for confidence.
   - Output: `vllm/outputs/result*_run*.jsonl`

2) **Score extraction + merge back to the original panel** (`vllm/json2csv.py`)
   - Parses JSONL to extract Return Movement Score and rationale.
   - Merges with the original question/features table by `(permno, year, month)`.
   - Output: `vllm/outputs/parsed_*.csv`

3) **Asset ranking/selection + portfolio construction/backtests** (`portfolio/main.py`)
   - Consumes `parsed_result(_except_gics)_runN.csv`.
   - Performs portfolio backtests including:
     - Long-only (equal-weight / value-weight)
     - Mean-Variance portfolios (Maximum Sharpe Ratio / Minimum Risk / Maximum Return)
   - Output: `portfolio/outputs/*.csv`

## Repository Layout

- `vllm/run.py`: question CSV → vLLM inference → `vllm/outputs/result*_run*.jsonl`
- `vllm/json2csv.py`: parse `result*.jsonl` + merge features → `vllm/outputs/parsed_*.csv`
- `portfolio/main.py`: backtests → `portfolio/outputs/*.csv`

## Environment (Conda)

```bash
conda env create -f environment.yml
conda activate sector
```

`environment.yml` is a minimal, maintainable spec (top-level dependencies only).

For exact reproducibility, `environment.lock.yml` is a full `conda env export` of the author's `vllm` environment (it includes transitive packages such as CUDA-related dependencies and may use non-default channels).

```bash
conda env create -f environment.lock.yml
conda activate sector
```

## Data Files

Default input files (relative paths):

- `data/llm_questions_2012_2021.csv`
  - LLM input table containing per-asset monthly prompts (includes `question` column).
- `data/question_features_2011_2021.csv`
  - Source question/features table used to merge model outputs back to the panel (merge key: `permno, year, month`).
- `data/prices_2012_2021.csv`
  - Price table used for backtests (uses `Date` column).
- `data/permno_monthly_meta.csv`
  - Permno-month metadata merged on `permno, date` (includes `prc/ret/shrout`, etc.).

See `data/README.md` for details.

## Running

```bash
# 1) LLM inference (outputs: vllm/outputs/)
python vllm/run.py --data-dir data --output-dir vllm/outputs

# 2) Parse/merge JSONL outputs into CSV (default: vllm/outputs/*.jsonl)
python vllm/json2csv.py --input-dir vllm/outputs --question-csv data/question_features_2011_2021.csv

# 3) Backtests (outputs: portfolio/outputs/)
python portfolio/main.py
```

## Prompt Variants (With vs. Without Sector Context)

`vllm/run.py` runs two prompt variants by default:

- **With sector context**: sector-specific analyst role using GICS (sector-aware prompting)
- **Without sector context**: analyst role without GICS conditioning

To use different input tables for the two variants, set `--csv-base` and `--csv-except` explicitly.

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
