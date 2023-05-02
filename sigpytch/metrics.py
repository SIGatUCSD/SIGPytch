import numpy as np
import pandas as pd

from periods import P

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

def max_drawdown(daily_returns: pd.Series, window: int = 252) -> pd.Series:
    """
    Computes the max drawdown for a given ticker.
    
    Parameters:
        daily_returns (pd.Series): Daily returns of the given ticker 
        window (int): Window size in trading days
    Returns:
        pd.Series: Max drawdown
    """
    # Compute drawdown based on max price in window
    roll_max = daily_returns.rolling(window, min_periods=1).max()
    drawdown = daily_returns/roll_max - 1.0
    max_drawdown = drawdown.rolling(window, min_periods=1).min()
    return max_drawdown

def return_average(price_data: pd.Series, window_width: int, window_index: int = 0) -> float:
    """
    Computes average returns over a period. O(1)
    
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
    Computes daily returns over a period O(window_width)
    
    Parameters:
        price_data (pd.Series): Daily price data
        window_index (int): Start index of the window
        window_width (int): Width of the window
    Returns:
        pd.Series: Daily returns over the provided window
    """
    if(window_index+window_width > len(price_data)):
        raise Exception("Invalid window width and/or index. Window outside price data series...")
    # TODO: currently using append which is amortized O(1), not true O(1). Better would be to allocate list of length window_width right away and then fill it, thus making loop true O(n) instead of amortized
    # TODO: even better, switch to pd.Series.apply() as it seems like it is more optimized https://towardsdatascience.com/400x-time-faster-pandas-data-frame-iteration-16fb47871a0a
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
    Computes volatility over some window of price data. O(window_width)
    
    Parameters:
        price_data (pd.Series): Daily price data
        window_width (int): Window size in trading days
        window_index (int): Window start trading day
    Returns:
        float: Volatility of the price over the given window
    """
    if(window_index+window_width > len(price_data)):
        raise Exception("Invalid window width and/or index. Window outside price data series...")
    
    av_returns = return_average(price_data, window_width, window_index)
    dly_returns = return_daily(price_data, window_width, window_index)
    T = window_width
    sigma = dly_returns.apply(lambda daily_return: ((daily_return - av_returns)**2)/T).sum()
    return sigma
    
def sharpe_ratio(price_data_asset: pd.Series, price_data_benchmark: pd.Series, window_width: int, window_index: int = 0) -> float:
    """
    Computes sharpe_ratio over a period between asset and benchamrk asset
    
    Parameters:
        price_data_asset (pd.Series): Daily price data of asset
        price_data_benchmark (pd.Series): Daily price data of benchmark asset
        window_index (int): Start index of the window
        window_width (int): Width of the window
    Returns:
        flaot: Sharpe ratio between asset and the benchmark asset
    """
    returns_asset = return_daily(price_data_asset, window_width, window_index)
    returns_benchmark = return_daily(price_data_benchmark, window_width, window_index)
    T = window_width
    D_t = pd.Series()
    for return_asset, return_benchmark in returns_asset, returns_benchmark:
        D_t.add(return_asset - return_benchmark)
    D_avr = D_t.mean()

    D_sigma = (D_t.apply(lambda D: (D-D_avr)**2).sum()/(T-1))**(1/2)
    return D_avr/D_sigma
    
