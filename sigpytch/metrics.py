import numpy as np
import pandas as pd

from .periods import P

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

def volatility(price_data: pd.Series, window_width: int, window_index: int = 0) -> float:
    av_returns = returns_average(price_data, window_width, window_index)
    dly_returns = returns_daily(price_data, window_width, window_index)
    T = len(dly_returns)
    sigma = 0
    for daily_return in dly_returns:
        sigma += ((daily_return - av_returns)**2)/T
    return sigma