from IPython.display import display


import os
os.environ["RD_LIB_CONFIG_PATH"] = "./"

import refinitiv.data as rd

import pandas as pd
import numpy as np

import refinitiv.data.eikon as ek


rd.open_session(config_name="./tests/refinitiv-data.config.json")

apple = rd.get_history(universe="AAPL.O")
display(apple)


peers = rd.get_data(universe=["AAPL.OQ", "Peers(AAPL.OQ)"], fields=["TR.PrimaryInstrument", "TR.F.MktCap"])
display(peers)

rd.close_session()