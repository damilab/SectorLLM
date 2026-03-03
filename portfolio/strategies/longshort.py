# longshort.py

import pandas as pd
import numpy as np
from utils import *

class LongShort:
    def __init__(self):
        pass

    # Select assets using absolute score thresholds.
    def select_absolute_threshold(self, df, end_date, long_threshold=0.7, short_threshold=0.7):
        df2 = df[df['magnitude_of_change'].isin(['Moderate', 'Large'])].copy()
        
        logprob_th = df2['logprobs'].quantile(0.0)
        df_lp = df2[df2['logprobs'] > logprob_th].copy()
        
        long_selected = df_lp[df_lp['return_movement'] >= long_threshold]
        short_selected = df_lp[df_lp['return_movement'] <= short_threshold]

        long_selected = long_selected.sort_values('return_movement', ascending=False)
        short_selected = short_selected.sort_values('return_movement', ascending=True)
        long_selected['portfolio_date'] = end_date
        short_selected['portfolio_date'] = end_date
        return long_selected, short_selected

    # Select assets using percentile thresholds (per month).
    def select_percentile_threshold(self, df, end_date, long_percentile=0.7, short_percentile=0.7):
        """
        Portfolio selection using percentile thresholds.
        """
        df2 = df[df['magnitude_of_change'].isin(['Moderate', 'Large'])].copy()
        
        long_th = df2['return_movement'].quantile(long_percentile / 100)
        short_th = df2['return_movement'].quantile(short_percentile / 100)

        long_selected = df2[df2['return_movement'] >= long_th]
        short_selected = df2[df2['return_movement'] <= short_th]

        long_selected = long_selected.sort_values('return_movement', ascending=False)
        short_selected = short_selected.sort_values('return_movement', ascending=True)
        long_selected['portfolio_date'] = end_date
        short_selected['portfolio_date'] = end_date
        return long_selected, short_selected
    
    def select_pct_logprob(self, df, end_date, n):
        #df2 = df#[df['magnitude_of_change'].isin(['Moderate', 'Large'])] 
        #df = df[df['return_movement'] >= 0.5].copy()
        
        #cut = df2['logprobs'].quantile(1 - pct / 100)
        #sel  = df2[df2['logprobs'] >= cut].copy()
        
        sel = df.sort_values("logprobs", ascending=False).head(n).copy()
        sel['portfolio_date'] = end_date
        #print(f"Selected {len(sel)} stocks with logprobs >= {cut:.4f} (pct: {pct})")
        return sel
    
    def select_pct_risk(self, df, end_date, n):
        #df2 = df#[df['magnitude_of_change'].isin(['Moderate', 'Large'])] 
        #df = df[df['return_movement'] >= 0.5].copy()
        
        #cut = df2['risk'].quantile(pct / 100)
        #sel  = df2[df2['risk'] >= cut].copy()
        sel = df.sort_values(['risk', 'logprobs'], ascending=[True, False]).head(n).copy()
        #sel = df.sort_values("risk", ascending=True).head(n).copy()
        sel['portfolio_date'] = end_date
        return sel

    def select_pct_movement(self, df, end_date, n):
        #df2 = df[df['magnitude_of_change'].isin(['Moderate', 'Large'])]
        
        #df2 = df2[df2['logprobs'] >=  df2['logprobs'].quantile(0.4)].copy()
        # cut = df2['return_movement'].quantile(1 - pct / 100)
        # sel = df2[df2['return_movement'] >= cut].copy()
        
        #sel = df.sort_values("return_movement", ascending=False).head(n).copy()
        sel = df.sort_values(['return_movement', 'perplexity'], ascending=[False, True]).head(n).copy()
        #sel = df.sort_values('proba', ascending=False).head(n).copy()
        #sel = df.sort_values('mom2m_avg', ascending=True).head(n).copy()
        sel['portfolio_date'] = end_date
        return sel

    def select_n_random(self, df, end_date, n, seed):
        sel = df.sample(n=min(n, len(df)), random_state=seed).copy()
        sel['portfolio_date'] = end_date
        return sel

def run_backtest_longshort(
    stock_data, df, rf, frequency, 
    selection_method: str = "return_movement",          # absolute | percentile | logprob | return_movement | risk | random_n
    # for selection_method="absolute"
    long_threshold: float = 0.5,
    short_threshold: float = 0.5,
    # for selection_method="percentile"
    long_percentile: int = 90,
    short_percentile: int = 10,
    # for selection_method in {"logprob","return_movement","risk"}:
    # NOTE: in the current implementation, `pct` behaves as top-N (count), not a percentile (uses head(n)).
    pct: float = 20.0,
    # for selection_method="random_n"
    n_random: int = 20,
    seed: int = 42,
    long_only=False):

    strategy = LongShort()
    calc = SharpeCalculator(rf)
    strategies       = ["equal", "value"]
    portfolio_types  = ["Long", "Short", "Long-Short"]

    period_metrics = {
        s: {p: {"return": [], "std": [], "sharpe": [], "mdd": [], "turnover": []}
            for p in portfolio_types}
        for s in strategies
    }
    daily_series = {s: {p: [] for p in portfolio_types} for s in strategies}
    weights_v    = {s: {p: [] for p in portfolio_types} for s in strategies}
    returns_v    = {s: {p: [] for p in portfolio_types} for s in strategies}

    counts = {s: [] for s in ["equal", "value"]}
    
    # Rebalancing loop
    rebalancing_dates = get_rebalancing_dates(stock_data, frequency)
    df['date'] = pd.to_datetime(df['date']).dt.normalize()

    for i in range(len(rebalancing_dates) - 2):
        insample, outsample = get_is_oos(stock_data, rebalancing_dates, i)
        # print(f"[{i}] insample: {insample.index.min()} ~ {insample.index.max()}")
        # print(f"[{i}] outsample: {outsample.index.min()} ~ {outsample.index.max()}")
        
        insample_end  = rebalancing_dates[i + 1].normalize()
        
        #pred = df[df['date'].isin(outsample.index)][['ticker', 'magnitude_of_change', 'confidence_score', 'return_movement', 'logprobs']]
        insample_df  = df[df['date'].isin(insample.index)].copy()
        # print("▶ insample_df max date:", insample_df['date'].max())
        # print("▶ outsample min date:", outsample.index.min())

        # (1) Select portfolio constituents
        if selection_method == "absolute":
            long_sel, short_sel = strategy.select_absolute_threshold(
                insample_df, insample_end, long_threshold, short_threshold)

        elif selection_method == "percentile":
            long_sel, short_sel = strategy.select_percentile_threshold(
                insample_df, insample_end, long_percentile, short_percentile)

        elif selection_method == "logprob":
            long_sel  = strategy.select_pct_logprob(insample_df, insample_end, pct)
            short_sel = pd.DataFrame(columns=long_sel.columns)   # no short leg

        elif selection_method == "return_movement":
            long_sel  = strategy.select_pct_movement(insample_df, insample_end, pct)
            short_sel = pd.DataFrame(columns=long_sel.columns)
            
        elif selection_method == "risk":
            long_sel  = strategy.select_pct_risk(insample_df, insample_end, pct)
            short_sel = pd.DataFrame(columns=long_sel.columns)   # no short leg

        elif selection_method == "random_n":
            long_sel  = strategy.select_n_random(insample_df, insample_end, n_random, seed)
            short_sel = pd.DataFrame(columns=long_sel.columns)

        # (2) Backtest per weighting scheme (equal / value)
        for weight_strategy in strategies:
            EMPTY_WDF = pd.DataFrame(columns=["ticker", "weight"])
            # Long weights
            if long_sel.empty:
                long_w_df = EMPTY_WDF.copy()
            else:
                long_w_df = apply_weight_strategy(long_sel, weight_strategy)

            # Short weights
            if short_sel.empty:                      
                short_w_df = EMPTY_WDF.copy()
            else:
                short_w_df = apply_weight_strategy(short_sel, weight_strategy)
    
            # Combine long and short legs
            ls_w_df            = long_w_df.copy()
            ls_w_df['weight'] -= short_w_df['weight']  
            
            # # Save weights (for turnover; currently disabled)
            w_long = long_w_df.set_index('ticker')['weight']
            # w_short = short_w_df.set_index('ticker')['weight']
            # w_ls    = w_long.sub(w_short, fill_value=0.0).rename('weight').reset_index()

            # weights_v[weight_strategy]['Long'].append(w_long)
            # weights_v[weight_strategy]['Short'].append(w_short)
            # weights_v[weight_strategy]['Long-Short'].append(w_ls)

            # Daily returns
            daily_long  = compute_portfolio_daily_returns(outsample, long_w_df,  sign=1)
            daily_short = compute_portfolio_daily_returns(outsample, short_w_df, sign=-1)
            daily_ls    = daily_long + daily_short

            # Segment metrics
            for ptype, series in zip(portfolio_types, [daily_long, daily_short, daily_ls]):
                rets, stds, sharps = compute_segment_metrics(
                    series,
                    [outsample.index[0], outsample.index[-1]],
                    rf
                )
                period_metrics[weight_strategy][ptype]['return'].extend(rets)
                period_metrics[weight_strategy][ptype]['std'].extend(stds)
                period_metrics[weight_strategy][ptype]['sharpe'].extend(sharps)
                daily_series[weight_strategy][ptype].append(series)

            # # Period returns (for turnover; currently disabled)
            # common_long  = w_long.index.intersection(outsample.columns)
            # common_short = w_short.index.intersection(outsample.columns)
            # common_ls    = w_ls.index.intersection(outsample.columns)

            # r_long  = outsample[common_long ].add(1).prod() - 1
            # r_short = outsample[common_short].add(1).prod() - 1
            # r_ls    = outsample[common_ls   ].add(1).prod() - 1

            # returns_v[weight_strategy]['Long'].append(r_long)
            # returns_v[weight_strategy]['Short'].append(r_short)
            # returns_v[weight_strategy]['Long-Short'].append(r_ls)

    # # Optional: turnover / MDD summaries
    # #print("\n=== Long/Short Turnover ===")
    # for w in strategies:
    #     for p in portfolio_types:
    #         tv = compute_turnover(weights_v[w][p], returns_v[w][p])
    #         #print(f"{w:<6} / {p:<10} turnover: {tv:.4f}")

    # #print("\n=== Long/Short MDD (from log rets) ===")
    # for w in strategies:
    #     for p in portfolio_types:
    #         mdd = compute_mdd_from_logrets(pd.concat(daily_series[w][p]))
    #         #print(f"{w:<6} / {p:<10} MDD: {mdd:.4f}")

            counts[weight_strategy].append(len(w_long))
            long_selected_tickers = long_sel['ticker'].tolist() if not long_sel.empty else []
    
    return period_metrics, daily_series, counts, long_selected_tickers
