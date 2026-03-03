# mean_variance.py  ─────────────────────────────────────────────
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from utils import *
from strategies.longshort import LongShort  

class MeanVarianceStrategy:
    def __init__(self, rf):
        self.rf = rf

    # Objective functions
    def max_sharpe_ratio(self, w, mu, cov):
        ret = np.dot(w, mu)
        std = np.sqrt(np.dot(w.T, np.dot(cov, w)))
        return - (ret - self.rf) / std

    def min_risk(self, w, cov):
        return np.sqrt(np.dot(w.T, np.dot(cov, w)))

    def max_return(self, w, mu):
        return -np.dot(w, mu)  # negate for minimization
    
    # Optimization
    def get_mvp_weights(self, insample):
        mu  = insample.mean() * 252             # annualized return
        cov = insample.cov()  * 252             # annualized covariance
        n   = len(mu)                           # number of assets
        w0 = np.ones(n) / n                     # initial weights
        bounds = [(0, 1) for _ in range(n)]     # long-only bounds
        constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1} # sum(weights) == 1
        
        mu_insample = mu.values
        cov_insample = cov.values

        weights_sharpe = minimize(self.max_sharpe_ratio, w0, args=(mu_insample, cov_insample),
                                method='SLSQP', bounds=bounds, constraints=constraints,
                                options={'maxiter': 100}).x
        
        weights_risk     = minimize(self.min_risk, w0, args=(cov_insample,),
                            method='SLSQP', bounds=bounds, constraints=constraints,
                            options={'maxiter': 100}).x
        
        weights_return    = minimize(self.max_return, w0, args=(mu_insample,),
                            method='SLSQP', bounds=bounds, constraints=constraints,
                            options={'maxiter': 100}).x
        
        return {
            "Maximum Sharpe Ratio": weights_sharpe,
            "Minimum Risk":        weights_risk,
            "Maximum Return":      weights_return
        }

def run_backtest_meanvar(
    stock_data, df, rf,
    frequency = "monthly",
    selection_method = "return_movement", # all | logprob | return_movement | risk | random_n
    # NOTE: `pct` is used as top-N (count), not a percentile, because selection uses head(n).
    pct = 20.0,
    n_random = 20, seed = 42):

    ls_strategy  = LongShort()          
    mv_strategy  = MeanVarianceStrategy(rf)
    calc         = SharpeCalculator(rf)
    strategy_names = ["Maximum Sharpe Ratio", "Minimum Risk", "Maximum Return"]

    performance_strategy  = {p: {"return": [], "std": [], "sharpe": []} for p in strategy_names}
    daily_returns = {p: [] for p in strategy_names}
    weights_all   = {p: [] for p in strategy_names}
    period_returns   = {p: [] for p in strategy_names}

    rebalancing_dates = get_rebalancing_dates(stock_data, frequency) # rebalancing dates
    
    df['date'] = pd.to_datetime(df['date']).dt.normalize()

    for i in range(len(rebalancing_dates) - 2):
        insample, outsample = get_is_oos(stock_data, rebalancing_dates, i)
        insample_end = rebalancing_dates[i+1].normalize()
        # if i == 0:
        #     print(f"[First rebalance] insample: {insample.index.min()} ~ {insample.index.max()}")
        #     print(f"[First rebalance] outsample: {outsample.index.min()} ~ {outsample.index.max()}")

        if selection_method == "all":
            tickers = insample.columns.tolist()
        else:
            insample_df = df[df['date'].isin(insample.index)].copy()
            #print(f"▶ [DEBUG] i={i}, insample_df.shape = {insample_df.shape}, selection_method = {selection_method}")
            if selection_method == "return_movement":
                sel = ls_strategy.select_pct_movement(insample_df, insample_end, pct)
            elif selection_method == "logprob":
                sel = ls_strategy.select_pct_logprob(insample_df, insample_end, pct)
            elif selection_method == "risk":
                sel = ls_strategy.select_pct_risk(insample_df, insample_end, pct)
            elif selection_method == "random_n":
                sel = ls_strategy.select_n_random(insample_df, insample_end, n_random, seed)

            tickers = sel['ticker'].tolist()
            #print(f"▶ [DEBUG] i={i}, Level={pct}, tickers count = {tickers}")
            if len(tickers) == 0:
                print(f"[WARN] i={i} tickers empty; skipping this period.")
                continue
            
        # if len(tickers) == 0:
        #     continue

        
        insample_sub = insample[tickers]
        outsample_sub = outsample[tickers]
        
        #print(insample_sub.shape, outsample_sub.shape, tickers)
        #print(insample_sub.columns.tolist(), outsample_sub.columns.tolist())

        
        # (2) Optimize weights for three mean-variance portfolios
        weight_mvp = mv_strategy.get_mvp_weights(insample_sub)
        #print(weight_mvp)

        # (3) Store segment metrics
        for p in strategy_names:
            w = pd.Series(weight_mvp[p], index=tickers)
            weights_all[p].append(w)

            # # Period returns (turnover) & daily returns
            # r_period = outsample_sub[tickers].add(1).prod() - 1
            # period_returns[p].append(r_period)

            daily = outsample_sub @ w.values
            rets, stds, sharps = compute_segment_metrics(
                daily, [outsample_sub.index[0], outsample_sub.index[-1]], rf)

            performance_strategy[p]["return"].extend(rets)
            performance_strategy[p]["std"].extend(stds)
            performance_strategy[p]["sharpe"].extend(sharps)
            daily_returns[p].append(daily)

    # # (4) Optional: turnover / MDD summaries
    # #print("\n=== Mean-Variance Turnover ===")
    # for p in strategy_names:
    #     tv = compute_turnover(weights_all[p], period_returns[p])
    #     #print(f"{p:<20} : {tv:.4f}")

    # #print("\n=== Mean-Variance MDD (from log-rets) ===")
    # for p in strategy_names:
    #     all_logs = pd.concat([np.log1p(s) for s in daily_returns[p]])
    #     mdd = compute_mdd_from_logrets(all_logs)
    #     #print(f"{p:<20} : {mdd:.4f}")

    period_metrics = {"mvp": performance_strategy}
    daily_series   = {"mvp": daily_returns}
    return period_metrics, daily_series, weights_all, period_returns, tickers
