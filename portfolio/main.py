# main.py

import pandas as pd
import numpy as np
import os
import multiprocessing
from pathlib import Path
import sys

# Ensure local imports work regardless of the current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import *
from strategies.longshort import run_backtest_longshort
from strategies.mean_variance import run_backtest_meanvar
from strategies.buy_hold import run_backtest_buyhold

# Paths (repo-relative; no absolute `/home/...` dependencies)
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
VLLM_OUTPUT_DIR = REPO_ROOT / "vllm" / "outputs"
PORTFOLIO_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

STOCK_PATH = DATA_DIR / "prices_2012_2021.csv"
SHROUT_PATH = DATA_DIR / "permno_monthly_meta.csv"


def summarize_long(pm, cnts, daily, rf, scheme):
    calc = SharpeCalculator(rf)
    met = pm[scheme]["Long"]
    dlist = daily[scheme]["Long"]

    m1_ret, m1_std, m1_shp = calc.method1(met)
    m2_ret, m2_std, m2_shp = calc.method2(np.concatenate([s.values for s in dlist]))
    m3_ret, m3_std, m3_shp, m3_avg_shp = calc.method3(met)
    return {
        "N": round(np.mean(cnts[scheme])),
        "M1_Ret": m1_ret,       "M1_Std": m1_std,       "M1_Sharpe": m1_shp,
        "M2_Ret": m2_ret,       "M2_Std": m2_std,       "M2_Sharpe": m2_shp,
        "M3_Ret": m3_ret,       "M3_Std": m3_std,       "M3_Sharpe": m3_shp,
        "M3_AvgSharpe": m3_avg_shp
    }


def summarize_mvp(mv_met, mv_daily, rf, level=None, method=None):
    level = "All" if level is None else level
    calc = SharpeCalculator(rf)
    rows = []
    for p, m in mv_met["mvp"].items():
        m1_r, m1_s, m1_sr = calc.method1(m)
        m2_r, m2_s, m2_sr = calc.method2(np.concatenate([s.values for s in mv_daily["mvp"][p]]))
        m3_r, m3_s, m3_sr, m3_avg = calc.method3(m)
        rows.append({
            "SelMethod": method,
            "Level": level,
            "MV_Port": p,
            "M1_Ret": m1_r, "M1_Std": m1_s, "M1_SR": m1_sr,
            "M2_Ret": m2_r, "M2_Std": m2_s, "M2_SR": m2_sr,
            "M3_Ret": m3_r, "M3_Std": m3_s, "M3_SR": m3_sr, "M3_Avg_SR": m3_avg
        })
    return rows

def concat_daily_to_monthly(daily_list):
    daily = pd.concat(daily_list)                 # list -> a single Series
    monthly = (1+daily).resample("M").prod()-1    # monthly compounded return
    return monthly


def run_one_job(args):
    """
    Worker function for multiprocessing.
    args = (run_idx, except_gics)
    """
    run_idx, except_gics = args
    suffix = "_except_gics" if except_gics else ""

    df_file = f"parsed_result{suffix}_run{run_idx}.csv"
    mv_file = f"mv_all{suffix}_2012_run{run_idx}.csv"
    lo_file = f"longonly_all{suffix}_2012_run{run_idx}.csv"


    df_path = VLLM_OUTPUT_DIR / df_file
    mv_output_path = PORTFOLIO_OUTPUT_DIR / mv_file
    lo_output_path = PORTFOLIO_OUTPUT_DIR / lo_file

    variant_label = "EXCEPT_GICS" if except_gics else "WITH_GICS"
    print(f"[PID {os.getpid()}] === Run {run_idx} | {variant_label} ===")

    # Load data
    df, prices, rf = get_data(
        df_path,
        STOCK_PATH,
        SHROUT_PATH,
        start_date='2012-01-01',
        end_date='2021-12-31'
    )

    # # Optional: Buy & Hold baseline
    # bh_met, bh_daily = run_backtest_buyhold(prices, df, rf, frequency="monthly")
    # print_results(compute_performance_metrics(bh_met, bh_daily, rf))

    # Experiment parameters
    # NOTE: In strategy code, `pct` currently behaves as top-N (count), not a percentile (uses head(n)).
    top_n_list = [2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    tasks = [("return_movement", top_n_list, "pct", {})]

    mv_rows, lo_rows, mv_mdd_rows = [], [], []
    
    for method, levels, arg_key, extra in tasks:
        for lv in levels:
            # Mean-Variance backtest
            mv_met, mv_daily, mv_weights, mv_period_rets, mv_tickers = run_backtest_meanvar(
                prices,
                df,
                rf,
                frequency="monthly",
                selection_method=method,
                **{arg_key: lv},
                **extra
            )
            
            # Summary (without MDD/turnover)
            mv_rows.extend(summarize_mvp(mv_met, mv_daily, rf, level=lv, method=method))
            
            # # Optional: summary including MDD & turnover
            # mv_mdd_rows.extend(create_performance_table_with_mdd_turnover(
            #     mv_met, mv_daily, mv_weights, mv_period_rets, rf, level=lv, method=method
            # ))

            # Long-only backtest (equal/value weighting)
            for scheme in ("equal", "value"):
                pm, dly, cnt, lo_tickers = run_backtest_longshort(
                    prices,
                    df,
                    rf,
                    frequency="monthly",
                    selection_method=method,
                    long_only=True,
                    **{arg_key: lv},
                    **extra
                )
                lo_rows.append({
                    "Weight": scheme,
                    "SelMethod": method,
                    "Level": lv,
                    **summarize_long(pm, cnt, dly, rf, scheme)
                })
                print(f"[run{run_idx} | {method} | top_n={lv} | {scheme}] "
                    f"MV vs LO tickers identical? {set(mv_tickers) == set(lo_tickers)}")

    # Save results
    pd.DataFrame(mv_rows).to_csv(mv_output_path, index=False)
    pd.DataFrame(lo_rows).to_csv(lo_output_path, index=False)

    print(f"[PID {os.getpid()}] [run{run_idx}] saved: MV -> {mv_output_path}, LO -> {lo_output_path}")


if __name__ == "__main__":
    # Run indices and variants
    run_indices = list(range(1, 101))
    variants = [True, False]
    jobs = [(v, f) for v in run_indices for f in variants]

    # Spawn one worker per CPU core
    cpu_count = os.cpu_count() or 1
    print(f"Main PID={os.getpid()}, cpu_cores={cpu_count}")

    with multiprocessing.Pool(processes=cpu_count) as pool:
        pool.map(run_one_job, jobs)
