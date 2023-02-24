# SIGPytch
Quantitative models and metrics used in the SIG at UCSD quarterly stock pitches.
## Install
`pip install sigpytch`
## Requirements
This project requires the following libraries:
- [numpy](https://numpy.org/news/#releases)
- [pandas](https://pandas.pydata.org/)
## Models
### Annualized Rolling Sharpe
`rolling_sharpe(daily_returns: pd.Series, risk_free_rate: pd.Series, window: int)`