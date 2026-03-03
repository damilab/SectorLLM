## buy_hold

import pandas as pd
import numpy as np
from utils import *

def run_backtest_buyhold(stock_data, df, rf, frequency="yearly"):
    """
    Run a Buy & Hold backtest.
    """
    # Create a portfolio using the first available row per ticker.
    portfolio_df = df.sort_values('date').groupby('ticker').first().reset_index()

    # # Use only tickers available in the yfinance price table
    # available_tickers = set(stock_data.columns)
    # portfolio_df = portfolio_df[portfolio_df['ticker'].isin(available_tickers)] # filter valid tickers

    # tickers = portfolio_df['ticker'].unique().tolist() # unique tickers
    # stock_data_subset = stock_data[tickers] # subset price data
    # portfolio_df = portfolio_df.drop_duplicates(subset=['ticker']) # drop duplicate tickers
    
    portfolio_types = ["equal", "value"]
    
    # Compute portfolios for each weighting scheme (equal, value).
    portfolios = {}
    for ptype in portfolio_types:
        portfolios[ptype] = apply_weight_strategy(portfolio_df, weight_strategy=ptype)

    daily_data = stock_data.pct_change().iloc[1:]  # daily returns
    daily_returns = {}
    for ptype in portfolio_types:
        w = portfolios[ptype].set_index('ticker')['weight']
        # Matrix multiply in column order
        daily_returns[ptype] = daily_data[w.index].fillna(0).dot(w.values)
    
    rebalancing_dates = get_rebalancing_dates(stock_data, frequency) # for segment metrics

    period_metrics = {p: {"return": [], "std": [], "sharpe": []} for p in portfolio_types}  # segment metrics
    daily_returns_dict = {p: [] for p in portfolio_types}  # daily return series

    # Segment metrics
    for port_type in portfolio_types:
        returns, stds, sharpes = compute_segment_metrics(daily_returns[port_type], rebalancing_dates, rf)
        period_metrics[port_type]["return"].extend(returns)
        period_metrics[port_type]["std"].extend(stds)
        period_metrics[port_type]["sharpe"].extend(sharpes)

        daily_returns_dict[port_type] = daily_returns[port_type]

    # Period metrics payload
    period_metrics = {
        "buy_hold": {
            "Equal": {
                "return": period_metrics["equal"]["return"],
                "std": period_metrics["equal"]["std"],
                "sharpe": period_metrics["equal"]["sharpe"]
            },
            "Value": {
                "return": period_metrics["value"]["return"],
                "std": period_metrics["value"]["std"],
                "sharpe": period_metrics["value"]["sharpe"]
            }
        }
    }

    # Daily return series payload
    daily_series = {
        "buy_hold": {
            "Equal": [daily_returns_dict["equal"]],
            "Value": [daily_returns_dict["value"]]
        }
    }

    # print("\n=== Buy&Hold MDD ===")
    # for ptype, series_list in daily_series["buy_hold"].items():
    #     # series_list = [pd.Series(arithmetic daily returns)]
    #     arith = series_list[0]
    #     # 1) arithmetic -> log returns
    #     logrets = np.log1p(arith)
    #     # 2) compute max drawdown
    #     mdd = compute_mdd_from_logrets(logrets)
    #     print(f"{ptype:<6} MDD: {mdd:.4f}")
    # ----------------------------------------------------------------

    return period_metrics, daily_series
