import sys

sys.path.append('../sigpytch')

from sigpytch.periods import *
import sigpytch.metrics as metrics

if __name__ == '__main__':
    beta = metrics.beta(ticker="MSFT")
    assert type(beta) == float
    assert beta is not None
    
    alpha = metrics.alpha(ticker="MSFT")
    assert type(alpha) == float
    assert alpha is not None