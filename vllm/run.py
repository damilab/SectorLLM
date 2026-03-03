"""
vLLM inference runner.

This script reads a "question CSV" (tabular prompts per permno/month), runs a chat LLM via vLLM,
and writes predictions to JSONL files under `vllm/outputs/` by default.

Downstream:
- `vllm/json2csv.py` parses these JSONL files into `parsed_*.csv`
- `portfolio/main.py` consumes `parsed_result(_except_gics)_runN.csv` for backtests
"""

import os
import json
import time
import gc
import yaml
import numpy as np
import pandas as pd
from vllm import LLM, SamplingParams
import warnings
import torch

warnings.filterwarnings("ignore")

import argparse
from pathlib import Path

# Load prompts from YAML
VLLM_DIR = Path(__file__).resolve().parent
with open(VLLM_DIR / "prompts.yaml", "r", encoding="utf-8") as f:
    PROMPTS = yaml.safe_load(f)

acronym_descriptions = PROMPTS["acronym_descriptions"].strip()
acronym_ex_descriptions = PROMPTS["acronym_ex_descriptions"].strip()
messages_template = PROMPTS["messages_template"]
messages_template_except_gics = PROMPTS["messages_template_except_gics"]

parser = argparse.ArgumentParser()
parser.add_argument("--gpu-memory-utilization", type=float, default=0.97)
parser.add_argument("--enable-chunked-prefill", action="store_true")  
parser.add_argument("--max-num-batched-tokens", type=int, default=40000)
parser.add_argument("--enable_prefix_caching", type=int, default=1)  # default enabled
parser.add_argument("--batch-size", type=int, default=800)  # batch size (tunable)
parser.add_argument("--data-dir", type=str, default=None, help="Directory containing input CSVs (default: <repo>/data)")
parser.add_argument("--output-dir", type=str, default=None, help="Directory to write JSONL outputs (default: <repo>/vllm/outputs)")
parser.add_argument("--csv-base", type=str, default=None, help="Input CSV for base run (default: <data-dir>/llm_questions_2012_2021.csv)")
parser.add_argument("--csv-except", type=str, default=None, help="Input CSV for except_gics run (default: <data-dir>/llm_questions_2012_2021.csv)")
parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
parser.add_argument("--num-runs", type=int, default=100, help="Number of inference runs (default: 100)")
args = parser.parse_args()

# Initialize the model once.
def initialize_model():
    llm = LLM(
        model=args.model,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_chunked_prefill=args.enable_chunked_prefill,
        enable_prefix_caching=args.enable_prefix_caching,
        dtype="bfloat16",
    )
    return llm

# GPU memory monitoring
def get_gpu_memory_usage():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3  # GB
    return 0

# Dynamic batch sizing
def get_optimal_batch_size(base_batch_size, gpu_memory_gb=None):
    # gpu_memory_gb currently unused
    return base_batch_size

# Memory cleanup
def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

 

def clean_assistant_prefix(text: str) -> str:
    if text.strip().lower().startswith("assistant"):
        parts = text.strip().split("\n", 1)
        return parts[1].strip() if len(parts) > 1 else ""
    return text.strip()

def format_time(sec: float) -> str:
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m}m {s}s"

def make_record(row, out):
    text = clean_assistant_prefix(out.text)
    cum_logprob = out.cumulative_logprob
    count = len(out.token_ids)
    avg_logprob = cum_logprob / count
    perplexity = np.exp(-avg_logprob)
    return {
        "permno": row["permno"],
        "year": row["year"],
        "month": row["month"],
        "question": row["question"],
        "gics": row["gics"],
        "prediction": text,
        "cumulative_logprob": cum_logprob,
        "avg_logprob": avg_logprob,
        "perplexity": perplexity
    }

# Batch conversion helper
def prepare_batch_conversations(batch, tokenizer, messages_template):
    """Convert a batch of rows into chat-formatted conversations."""
    convs = []
    for _, row in batch.iterrows():
        messages = [
            {
                "role": m["role"],
                "content": m["content"].format(row=row, 
                                                acronym_descriptions=acronym_descriptions,
                                                acronym_ex_descriptions=acronym_ex_descriptions, ),
            }
            for m in messages_template
        ]
        convs.append(
            tokenizer.apply_chat_template(messages, tokenize=False)
        )
    return convs


def run_single_inference(llm, tokenizer, csv_file, messages_template, output_prefix, run_idx):
    """Run a single inference pass (model is passed in)."""
    print(f"GPU Memory before loading data: {get_gpu_memory_usage():.2f} GB")
    
    df = pd.read_csv(csv_file, parse_dates=["date"])
    df = df.query('"2021-12-31" >= date >= "2012-01-01"').reset_index(drop=True)

    total = len(df)
    
    # Determine batch size
    batch_size = get_optimal_batch_size(args.batch_size, None)
    
    processed_total = 0
    start_time = time.time()

    # Output path
    jsonl_path = f"{output_prefix}_run{run_idx}.jsonl"

    with open(jsonl_path, "w", encoding="utf-8") as fout:
        for start in range(0, total, batch_size):
            batch = df.iloc[start : start + batch_size]
            
            # Convert batch to prompts
            convs = prepare_batch_conversations(batch, tokenizer, messages_template)

            # Run inference
            outputs = llm.generate(
                convs,
                SamplingParams(
                    temperature=0,
                    max_tokens=512,
                    logprobs=1,
                    stop_token_ids=[
                        tokenizer.eos_token_id,
                        tokenizer.convert_tokens_to_ids("<|eot_id|>")
                    ],
                )
            )

            # Write results
            for out, (_, row) in zip(outputs, batch.iterrows()):
                rec = make_record(row, out.outputs[0])
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                processed_total += 1

            # Flush file buffer
            fout.flush()
        
            # Progress
            pct = processed_total / total
            elapsed = time.time() - start_time
            eta = elapsed / pct - elapsed if pct > 0 else 0
            throughput = processed_total / elapsed if elapsed > 0 else 0
            
            print(f"Batch size={batch_size}, GPU Memory: {get_gpu_memory_usage():.2f} GB, Throughput: {throughput:.1f} req/s")
            print(f"[Run {run_idx}] {processed_total}/{total} ({pct:.1%}) Elapsed: {format_time(elapsed)} | ETA: {format_time(eta)}")

    print(f"Completed run {run_idx}: {jsonl_path}")
    cleanup_memory()


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    vllm_dir = Path(__file__).resolve().parent

    data_dir = Path(args.data_dir) if args.data_dir else (repo_root / "data")
    output_dir = Path(args.output_dir) if args.output_dir else (vllm_dir / "outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the model once
    print("Initializing model...")
    llm = initialize_model()
    tokenizer = llm.get_tokenizer()
    print(f"Model loaded. GPU Memory: {get_gpu_memory_usage():.2f} GB")

    # Base variant
    csv_base = Path(args.csv_base) if args.csv_base else (data_dir / "llm_questions_2012_2021.csv")
    base_prefix = str(output_dir / "result")
    
    
    # except_gics variant
    csv_except = Path(args.csv_except) if args.csv_except else (data_dir / "llm_questions_2012_2021.csv")
    except_prefix = str(output_dir / "result_except_gics")

    if not csv_base.exists():
        raise FileNotFoundError(f"base csv not found: {csv_base}")
    if not csv_except.exists():
        raise FileNotFoundError(f"except csv not found: {csv_except}")
    if csv_base.resolve() == csv_except.resolve():
        print(f"[NOTE] csv_base and csv_except are the same file: {csv_base}")
    print(f"[PATH] data_dir={data_dir}")
    print(f"[PATH] output_dir={output_dir}")
    print(f"[PATH] csv_base={csv_base}")
    print(f"[PATH] csv_except={csv_except}")

    # Runs
    try:
        for run_idx in range(1, args.num_runs + 1):
            print(f"\n==== [WITH-GICS TEMPLATE] Run {run_idx} ====")
            run_single_inference(llm, tokenizer, csv_base, messages_template, base_prefix, run_idx)
            
            # Periodic cleanup (every 10 runs)
            if run_idx % 10 == 0:
                print(f"Periodic cleanup at run {run_idx}")
                cleanup_memory()
                
        for run_idx in range(1, args.num_runs + 1):
            print(f"\n==== [EXCEPT-GICS TEMPLATE] Run {run_idx} ====")
            run_single_inference(llm, tokenizer, csv_except, messages_template_except_gics, except_prefix, run_idx)
            
            # Periodic cleanup (every 10 runs)
            if run_idx % 10 == 0:
                print(f"Periodic cleanup at run {run_idx}")
                cleanup_memory()
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Final cleanup
        cleanup_memory()
        print("Final cleanup completed")
