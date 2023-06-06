import numpy as np
import pandas as pd
import yfinance as yf

from periods import *

def rolling_sharpe(daily_returns: pd.Series,
                   risk_free_rate: pd.Series, window: int) -> pd.Series:
    """
    Computes the rolling Sharpe ratio for a given ticker and risk-free rate.
    
    Parameters:
        daily_returns (pd.Series): Daily returns of the given ticker 
        risk_free_rate (pd.Series): Risk-free rate (treasury yield)
        window (int): Window size in months
    Returns:
        pd.Series: Rolling Sharpe ratio  
    """
    # Compute D-bar
    avg_return = daily_returns.rolling(window = window * P.TDAYS_PER_MONTH).mean()
    periodized_avg_return = avg_return * window * P.TDAYS_PER_MONTH
    avg_diff_return = periodized_avg_return - risk_free_rate
    # Compute sigma
    volatility = daily_returns.rolling(window = window * P.TDAYS_PER_MONTH).std()
    periodized_volatility = volatility * np.sqrt(window * P.TDAYS_PER_MONTH)
    # Compute rolling Sharpe ratio: D-bar / sigma
    rolling_sharpe_ratio = (avg_diff_return / periodized_volatility).dropna()
    return rolling_sharpe_ratio

def max_drawdown(close: pd.Series, window: int = 252) -> pd.Series:
    """
    Computes the max drawdown for a given ticker.
    
    Parameters:
        close (pd.Series): Daily closing price of the given ticker 
        window (int): Window size in trading days
    Returns:
        pd.Series: Max drawdown
    """
    # Compute drawdown based on max price in window
    roll_max = close.rolling(window, min_periods=1).max()
    drawdown = close/roll_max - 1.0
    max_drawdown = drawdown.rolling(window, min_periods=1).min()
    return max_drawdown

def calmar(daily_returns: pd.Series, risk_free_rate: pd.Series, window: int = 6) -> pd.Series:
    """
    Computes the rolling Calmar ratio for a given ticker.
    
    Parameters:
        daily_returns (pd.Series): Daily returns of the given ticker 
        window (int): Window size in months
    Returns:
        pd.Series: Rolling Calmar ratio
    """
    # Compute sigma
    avg_return = daily_returns.rolling(window = window * P.TDAYS_PER_MONTH).mean()
    periodized_avg_return = avg_return * window * P.TDAYS_PER_MONTH
    avg_diff_return = periodized_avg_return - risk_free_rate
    # Compute max drawdown
    max_draw = max_drawdown(daily_returns, window)
    # Compute rolling Calmar ratio: D-bar / max_drawdown
    calmars = (avg_diff_return/abs(max_draw)).dropna()
    return calmars

def return_average(price_data: pd.Series, window_width: int, window_index: int = 0) -> float:
    """
    Computes average returns over a period
    
    Parameters:
        price_data (pd.Series): Daily price data
        window_index (int): Start index of the window
        window_width (int): Width of the window
    Returns:
        float: Average return over a provided window
    """
    if(window_index+window_width > len(price_data)):
        raise Exception("Invalid window width and/or index. Window outside price data series...")
    return (price_data[window_index+window_width - 1] - price_data[window_index])/price_data[window_index]

def return_daily(price_data: pd.Series, window_width: int, window_index: int = 0) -> pd.Series:
    """
    Computes daily returns over a period
    
    Parameters:
        price_data (pd.Series): Daily price data
        window_index (int): Start index of the window
        window_width (int): Width of the window
    Returns:
        pd.Series: Daily returns over the provided window
    """
    if(window_index+window_width > len(price_data)):
        raise Exception("Invalid window width and/or index. Window outside price data series...")
    
    o = []

    prev_daily_price = -1
    for daily_price in price_data:
        if(prev_daily_price < 0):
            prev_daily_price = daily_price
            continue
        o.append((daily_price-prev_daily_price)/prev_daily_price)
    return pd.Series(data=o)

def volatility(price_data: pd.Series, window_width: int, window_index: int = 0) -> float:
    """
    Computes volatility over some window of price data
    
    Parameters:
        price_data (pd.Series): Daily price data
        window_width (int): Window size in trading days
        window_index (int): Window start trading day
    Returns:
        float: Volatility of the price over the given window
    """
    av_returns = return_average(price_data, window_width, window_index)
    dly_returns = return_daily(price_data, window_width, window_index)
    T = len(dly_returns)
    sigma = 0
    for daily_return in dly_returns:
        sigma += ((daily_return - av_returns)**2)/T
    return sigma
    
def rmse(obs: np.array, preds: np.array) -> float:
    """
    Calculates root mean squared error for two array-likes

    Parameters:
        obs: Array-like containing observed values
        preds: Array-like containing predictions
    Returns:
        float: Root mean squared error of the two array-likes
    """
    
    # calculates RMSE
    return np.sqrt(np.mean((obs - preds) ** 2))

def beta(obs: np.array = None, benchmark: np.array = None, ticker: str = None) -> float:
    """
    Calculates the beta ratio for a stock and a given index

    Parameters:
        obs (np.array): an array-like containing price data for the stock
        benchmark (np.array): an array-like containing price data for the index/benchmark
        ticker (str): Ticker symbol of a stock
    Returns:
        float: The beta ratio which measures volatilty of a stock relative to the market
    """
    
    # raises error if all args are None
    if obs is None:
        # loads monthly data of a ticker if no data is passed
        try:
            obs = yf.Ticker(ticker).history(start=DATE.START).Close[::INTERVALS.MONTHLY]
        except:
            raise ValueError("Must have a valid Ticker symbol")
        
    # loads monthly S&P 500 data as the benchmark by default
    if benchmark is None:
        benchmark = yf.Ticker("^GSPC").history(start=DATE.START).Close[::INTERVALS.MONTHLY]
    
    # loads data of a ticker if no data is passed
    if obs is None:
        obs = yf.Ticker(ticker).history(start=DATE.START).Close[::INTERVALS.MONTHLY]
    
    # standardizes data
    obs = obs.pct_change().dropna()
    benchmark = benchmark.pct_change().dropna()
    
    # calculates Beta
    return np.round(np.cov(obs, benchmark)[0, 1] / np.var(benchmark), 2)

def alpha(ticker: str = None, benchmark: np.array = None, rf: np.array = None) -> float:
    """
    Calculates the alpha ratio for a stock and a given index

    Parameters:
        obs (np.array): an array-like containing price data for the stock
        benchmark (np.array): an array-like containing price data for the index
        rf (np.array): an array-like containing data for the risk-free index
        ticker (str): a Ticker symbol
    Returns:
        float: The alpha ratio which measures excess returns of a stock relative to the market
    """
    
    # raises ValueError if no valid Ticker is passed
    if ticker is None:
        raise ValueError("Must be a valid Ticker symbol")
    
    # generates risk-free index data using 30-yr treasuries by default
    if rf is None:
        rf = yf.Ticker("^TYX").history(start=DATE.START).Close
        
    # loads monthly S&P 500 data as the benchmark by default
    if benchmark is None:
        benchmark = yf.Ticker("^GSPC").history(start=DATE.START).Close
        
    # generates stock data from Ticker symbol
    obs = yf.Ticker(ticker).history(start=DATE.START).Close
    
    # generates returns for the period for each array-like
    returns = obs.iloc[-1] / obs.iloc[0]
    
    rfr = rf.iloc[-1] / rf.iloc[0]
    
    index_returns = benchmark.iloc[-1] / benchmark.iloc[0]
    
    return returns - rfr - (beta(ticker="MSFT") * (index_returns - rfr))