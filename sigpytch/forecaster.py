"""
BSD 3-Clause License

Copyright (c) 2023, Sustainable Investment Group at UCSD

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

from periods import P

class LSTMForecaster():

    def __init__(self, series: pd.Series):
        self.__series = series
        self.model = None

        self.__lags =  P.TDAYS_PER_MONTH
        self.forecast_len = P.TDAYS_PER_YEAR

        self.__scaler = MinMaxScaler(feature_range=(-1, 1))

    @property
    def forecast_len(self) -> int:
        return self.forecast_len
    
    @forecast_len.setter
    def forecast_len(self, new_len: int):
        self.forecast_len = new_len

    def __setup_model(self, Y: np.ndarray):
        self.model = Sequential()
        self.model.add(LSTM(64, input_shape=(self.__lags, 1)))
        self.model.add(Dense(Y.shape[1])) # output layer
        self.model.compile(optimizer="adam", loss="mse")

    def __prepare_data(self, Y: np.ndarray):
        Y_scaled = self.__scaler(Y_scaled)
        
        n_fut = self.forecast_len
        n_past = self.__lags
        period = n_fut + n_past

        idx_end = len(Y_scaled)
        idx_start = idx_end - period

        X_prep = []
        Y_prep = []
        while idx_start > 0:
            X_prep.append([self.__scaled_data[idx_start:idx_start+n_past]])
            Y_prep.append(self.__scaled_data[idx_start+n_past:idx_start+period])
            idx_start -= 1

        X_prep = np.array(X_prep)
        X_prep = X_prep.reshape((X_prep.shape[0], X_prep.shape[1], 1))
        Y_prep = np.array(Y_prep)
        Y_prep = Y_prep.reshape((-1, 1))
        return X_prep, Y_prep
    
    def train(self, lags: int, epochs: int):
        self.__lags = lags
        
        TEST_SPLIT = 0.2
        test_len = int(len(self.__series) * TEST_SPLIT)
        train_set = self.__series[ :-test_len] # first 80%
        test_set = self.__series[-test_len: ] # last 20%

        X_train, Y_train = self.__prepare_data(train_set)
        if self.model is None: self.__setup_model(Y_train)

        self.model.fit(X_train, Y_train, epochs=epochs)
        




    

    

