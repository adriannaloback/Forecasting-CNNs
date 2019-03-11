import math
import numpy as np
import pandas as pd

class DataLoader():
    """
    A class for loading and transforming data for the cnn model.
    Note that cross validation (CV) to evaluate performance on
    validation data will be performed for each individual model
    by GridSearchCV (see run_wgridsearch.py).
    """

    def __init__(self, filename, cols):
        dataframe       = pd.read_csv(filename, delim_whitespace=True)
        self.data_train = dataframe.get(cols).values
        self.len_train  = len(self.data_train)
        self.len_train_windows = None

    def get_train_data(self, seq_len, normalise):
        '''
        Create x, y train data windows (sliding windows)
        Warning: batch method, not generative, make sure have enough
        memory to load data.
        '''
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            x, y = self._next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def _next_window(self, i, seq_len, normalise):
        '''
        Generates the next data window from the given index location i
        '''
        window = self.data_train[i:i+seq_len]
        window = self.normalise_windows(window, single_window=True)[0] if normalise else window
        x = window[:-1]
        y = window[-1, [0]] #assumes 0th col is target var to predict
        return x, y

    def normalise_windows(self, window_data, single_window=False):
        '''Normalise window with a base value of zero'''
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = []
            for col_i in range(window.shape[1]):
                normalised_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
                normalised_window.append(normalised_col)
            normalised_window = np.array(normalised_window).T # reshape and transpose array back into original multidimensional format
            normalised_data.append(normalised_window)
        return np.array(normalised_data)
