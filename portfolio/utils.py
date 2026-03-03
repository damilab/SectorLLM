## utils.py

import pandas as pd
import numpy as np
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

def get_data(df_path, stock_path, shrout_path,
             start_date='2012-01-01', end_date='2021-12-31',
             max_workers=20):
    """
    Load and merge data, load price table, and compute a risk-free rate.

    Notes:
    - Filters both the prediction table and the price table to the given date range.
    """

    # 1) Load prediction outputs (parsed CSV)
    df = pd.read_csv(df_path).drop_duplicates().sort_values(['date', 'permno'])
    # Parse `date`
    df['date'] = pd.to_datetime(df['date'])
    # Filter to requested window
    df = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]
    
    # Load shares-outstanding / metadata
    shrout = pd.read_csv(shrout_path).sort_values(['date', 'permno'])
    shrout['date'] = pd.to_datetime(shrout['date'])
    # Filter to requested window
    shrout = shrout[(shrout['date'] >= pd.to_datetime(start_date)) & (shrout['date'] <= pd.to_datetime(end_date))]
    
    # Merge (keep legacy column handling)
    df = df.drop(columns=['prc', 'ret', 'shrout', 'ep'], errors='ignore')
    df = df.merge(shrout, on=['date', 'permno']).drop_duplicates()
    df['market_cap'] = df['shrout'] * abs(df['prc'])
    df['ticker']     = df['ticker_yf']

    # Shift predictions to align with portfolio construction timing.
    cols_to_shift = ['return_movement',
                     'logprobs', 'perplexity'] # the stored predictions target the next period; shift to align in-sample vs out-of-sample
    df = df.sort_values(['ticker', 'date'])
    for col in cols_to_shift:
        df[col] = df.groupby('ticker')[col].shift(-1)

    # 2) Load price table and set index
    stock_data = pd.read_csv(stock_path).dropna(axis=1)
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data.set_index('Date', inplace=True)
    # Filter to requested window
    stock_data = stock_data.loc[start_date : end_date]

    # 3) Compute a risk-free rate (example: ^IRX)
    rf = yf.download('^IRX',
                     start=start_date,
                     end=end_date,
                     auto_adjust=True,
                     progress=False)['Close'].mean().iloc[0] / 100

    # 4) Keep only tickers that exist in both tables
    common_tickers = list(df['ticker'].unique())
    common_tickers = [t for t in common_tickers if t in stock_data.columns]
    df = df[df['ticker'].isin(common_tickers)]
    stock_data = stock_data[common_tickers]

    return df, stock_data, rf

def get_rebalancing_dates(stock_data, frequency='monthly'):
    """
    Compute rebalancing dates (monthly, quarterly, yearly).
    """
    if frequency == 'monthly': # monthly
        rebalancing_dates = stock_data.resample('M').last().index
    elif frequency == 'quarterly': # quarterly
        rebalancing_dates = stock_data.resample('Q').last().index
    elif frequency == 'yearly': # yearly
        monthly_index = stock_data.resample('M').last().index
        rebalancing_dates = monthly_index[monthly_index.month == 6]
    return rebalancing_dates

def get_is_oos(stock_data, rebalancing_dates, i):
    """
    Split returns into in-sample / out-of-sample windows per rebalancing period.
    """
    insample_start = rebalancing_dates[i]
    insample_end = rebalancing_dates[i+1]
    outsample_start = rebalancing_dates[i+1]
    outsample_end = rebalancing_dates[i+2]
    
    insample_data = stock_data.loc[insample_start:insample_end].pct_change().iloc[1:] # in-sample
    outsample_data = stock_data.loc[outsample_start:outsample_end].pct_change().iloc[1:] # out-of-sample
    combined_data = pd.concat([insample_data, outsample_data], axis=0).dropna(axis=1) # keep common tickers
    
    # Split again after dropping columns
    insample_data = combined_data.loc[insample_data.index]
    outsample_data = combined_data.loc[outsample_data.index]
    # print("is:", insample_data.head())
    # print("oos:", outsample_data.head())
    
    return insample_data, outsample_data

def apply_weight_strategy(portfolio_df, weight_strategy='equal'):
    """
    Assign portfolio weights (equal-weight or value-weight).
    """
    portfolio_df = portfolio_df.copy()
    
    if weight_strategy == 'equal':
        portfolio_df['weight'] = 1.0 / len(portfolio_df)
    elif weight_strategy == 'value':
        portfolio_df['weight'] = portfolio_df['market_cap'] / portfolio_df['market_cap'].sum()
        
    return portfolio_df

def calc_portfolio_metrics(w, mu, cov_matrix, rf):
    """
    Compute annualized return, risk, and Sharpe ratio.
    """
    ret = np.dot(w, mu)
    std = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
    sharpe = (ret - rf) / std
    return ret, std, sharpe

class SharpeCalculator:
    """
    Three ways to compute Sharpe ratio.
    """
    def __init__(self, rf: float):
        self.rf = rf

    def method1(self, metrics_dict):
        # Mean/std over per-period metrics
        ret = np.mean(metrics_dict["return"])
        std = np.std(metrics_dict["return"]) # std of per-period returns
        sharpe = (ret - self.rf) / std
        return ret, std, sharpe

    def method2(self, returns_array):
        # Using the full daily return series
        ret = np.mean(returns_array) * 252
        std = np.std(returns_array) * np.sqrt(252)
        sharpe = (ret - self.rf) / std
        return ret, std, sharpe

    def method3(self, metrics_dict):
        # Compute Sharpe using mean return and mean std over periods
        ret = np.mean(metrics_dict["return"])
        std = np.mean(metrics_dict["std"]) # mean of per-period std
        sharpe = (ret - self.rf) / std
        avg_sharpe = np.mean(metrics_dict["sharpe"]) # mean of per-period Sharpe values
        return ret, std, sharpe, avg_sharpe

def compute_portfolio_daily_returns(outsample_data, portfolio_df, sign=1):
    """
    Vectorized computation of daily portfolio returns from out-of-sample returns
    and a portfolio DataFrame.
    """
    # 1) Extract weights
    if portfolio_df.empty:
        return pd.Series(0.0, index=outsample_data.index)
    
    # Weight series
    weights = portfolio_df.set_index('ticker')['weight']

    # 2) Keep common tickers and fill NaNs with 0
    common_tickers = weights.index.intersection(outsample_data.columns)
    if common_tickers.empty:
        return pd.Series(0.0, index=outsample_data.index)
    
    arr = outsample_data[common_tickers].fillna(0).values  # shape (T, N)
    w   = weights.loc[common_tickers].values               # shape (N,)
    daily = arr @ w * sign

    return pd.Series(daily, index=outsample_data.index)

def compute_segment_metrics(daily_returns, rebalancing_dates, rf):
    """
    Compute segment metrics given daily returns and rebalancing boundaries.
    """
    returns, stds, sharpes = [], [], []
    for i in range(len(rebalancing_dates) - 1):
        start_date, end_date = rebalancing_dates[i], rebalancing_dates[i+1] # segment boundaries
        segment = daily_returns.loc[start_date:end_date] # segment daily returns
        
        ann_ret = np.mean(segment) * 252 # annualized return
        ann_std = np.std(segment) * np.sqrt(252) # annualized std
        ann_sharpe = (ann_ret - rf) / ann_std # Sharpe ratio

        returns.append(ann_ret)
        stds.append(ann_std)
        sharpes.append(ann_sharpe)
    
    return returns, stds, sharpes

def compute_performance_metrics(period_metrics, daily_series, rf):
    """
    Compute performance metrics (three methods).
    """
    calc = SharpeCalculator(rf)
    methods = {"Method 1": {}, "Method 2": {}, "Method 3": {}
    }
    
    # Method 1 and 3: per-period metrics
    for scheme, ports in period_metrics.items():
        methods["Method 1"][scheme] = {}
        methods["Method 3"][scheme] = {}
        for port, metrics in ports.items():
            ret1, std1, sharpe1 = calc.method1(metrics)
            methods["Method 1"][scheme][port] = {
                "Return": ret1,
                "Std": std1,
                "Sharpe Ratio": sharpe1
            }
            ret3, std3, sharpe3, avg_sharpe3 = calc.method3(metrics)
            methods["Method 3"][scheme][port] = {
                "Return": ret3,
                "Std": std3,
                "Sharpe Ratio": sharpe3,
                "Avg. Sharpe Ratio": avg_sharpe3
            }
    
    # Method 2: merge all daily returns
    for scheme, ports in daily_series.items():
        methods["Method 2"][scheme] = {}
        for port, series_list in ports.items():
            combined_daily = pd.concat(series_list) if isinstance(series_list[0], pd.Series) else np.array(series_list)
            ret2, std2, sharpe2 = calc.method2(combined_daily.values if isinstance(combined_daily, pd.Series) else combined_daily)
            methods["Method 2"][scheme][port] = {
                "Return": ret2,
                "Std": std2,
                "Sharpe Ratio": sharpe2
            }
    
    return methods

def print_results(results_dict):
    """
    Print results tables.
    """
    for method_name, method_results in results_dict.items():
        print(f"\n{method_name}:")
        for scheme, data in method_results.items():
            df_res = pd.DataFrame(data).T
            print(f"<{scheme}>")
            print(df_res, "\n")
