"""
Parse vLLM inference outputs (JSON/JSONL) into CSV and merge with the original question/features CSV.

Inputs (defaults, relative to this repo):
- vllm/outputs/result*_run*.jsonl : vLLM outputs created by vllm/run.py
- data/question_features_2011_2021.csv : original question/features table used for merge keys

Outputs:
- vllm/outputs/parsed_<input_basename>.csv

Notes:
- Merge key: (permno, year, month)
- Output columns are intentionally aligned to what portfolio/ code expects:
  return_movement, rationale, cumulative_logprob, logprobs(=avg_logprob), perplexity, ...
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

RETURN_MOVEMENT_SCORE_PATTERN = re.compile(
    r"(\*\*Return Movement Score:?\*\*(?::)?\s*(-?\d+(?:\.\d+)?))|"
    r"(\*\*Prediction for t\+1\*\*:?[\s\n]*(-?\d+(?:\.\d+)?))"
)
RATIONALE_PATTERN = re.compile(
    r"(\*\*Rationale(\*\*:?|:\*\*):? ?|-\s\*\*Rationale\*\*: ?)"
    r"(.*?)(\n- \*\*Magnitude of Change|\n\n|$)",
    re.S,
)


def iter_json_or_jsonl(fp):
    """Yield dicts from either a JSON list file or a JSONL file (line-delimited JSON)."""
    head = fp.read(2048)
    fp.seek(0)
    first = next((c for c in head if not c.isspace()), "")
    if first == "[":
        for obj in json.load(fp):
            yield obj
        return

    for line in fp:
        line = line.strip()
        if line:
            yield json.loads(line)


def load_question_table(question_csv: Path) -> pd.DataFrame:
    """Load the original question/features table and normalize merge keys."""
    question_df = pd.read_csv(question_csv, parse_dates=["date"])
    question_df["year"] = question_df["date"].dt.year
    question_df["month"] = question_df["date"].dt.month
    for col in ["permno", "year", "month"]:
        question_df[col] = pd.to_numeric(question_df[col], errors="coerce")
    return question_df


def parse_predictions(prediction_json_path: Path) -> pd.DataFrame:
    """Parse prediction JSON/JSONL into a flat table used for merge."""
    records = []
    with prediction_json_path.open("r", encoding="utf-8") as f:
        for obj in iter_json_or_jsonl(f):
            prediction_text = obj.get("prediction", "") or ""

            m = RETURN_MOVEMENT_SCORE_PATTERN.search(prediction_text)
            score_text = (m.group(2) or m.group(4)) if m else None

            rm = RATIONALE_PATTERN.search(prediction_text)
            rationale = rm.group(3).strip() if rm else None

            records.append(
                {
                    "permno": obj.get("permno"),
                    "year": obj.get("year"),
                    "month": obj.get("month"),
                    "return_movement": score_text,
                    "rationale": rationale,
                    "cumulative_logprob": obj.get("cumulative_logprob"),
                    "logprobs": obj.get("avg_logprob"),
                    "perplexity": obj.get("perplexity"),
                }
            )

    pred_df = pd.DataFrame(records).drop_duplicates()
    pred_df["return_movement"] = pd.to_numeric(
        pred_df["return_movement"]
        .astype(str)
        .str.extract(r"(-?\d+(?:\.\d+)?)", expand=False),
        errors="coerce",
    )
    for col in ["permno", "year", "month"]:
        pred_df[col] = pd.to_numeric(pred_df[col], errors="coerce")
    return pred_df


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_input_dir = Path(__file__).resolve().parent / "outputs"
    default_question_csv = repo_root / "data" / "question_features_2011_2021.csv"

    ap = argparse.ArgumentParser(description="Parse vLLM JSONL outputs and merge with question/features CSV.")
    ap.add_argument("--input-dir", type=str, default=str(default_input_dir), help="Directory with result*.json(l) files.")
    ap.add_argument("--question-csv", type=str, default=str(default_question_csv), help="Original question/features CSV.")
    ap.add_argument("--glob", dest="glob_pattern", type=str, default="*.jsonl", help="File glob under input-dir (default: *.jsonl)")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    question_csv = Path(args.question_csv)

    if not question_csv.exists():
        raise FileNotFoundError(f"question csv not found: {question_csv}")
    if not input_dir.exists():
        raise FileNotFoundError(f"input dir not found: {input_dir}")

    question_df = load_question_table(question_csv)

    paths = sorted(input_dir.glob(args.glob_pattern))
    if not paths:
        raise FileNotFoundError(f"no inputs matched: {input_dir / args.glob_pattern}")

    for prediction_path in paths:
        base_name = prediction_path.stem
        output_csv = input_dir / f"parsed_{base_name}.csv"

        pred_df = parse_predictions(prediction_path)
        merged_df = pred_df.merge(question_df, on=["permno", "year", "month"], how="inner")

        merged_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"Saved merged dataframe to {output_csv} (rows: {len(merged_df)})")


if __name__ == "__main__":
    main()
