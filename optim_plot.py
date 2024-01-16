from numpy.random import default_rng
import pandas as pd
from part2_house_value_regression import *
from sklearn.experimental import enable_halving_search_cv  # noqa # This is needed to confirm functionality of halvinggridsearch, an experimental scikit feature.

from sklearn.model_selection import HalvingGridSearchCV, KFold

import numpy as np
import matplotlib.pyplot as plt


def RegressorHyperParameterSearch(data, k=10):
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        - k {int} -- number of folds.
        
    Returns the best regressor.
    """
    
    output_label = "median_house_value"
    # Separate input from output:
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]

    shuffled_indices = default_rng().permutation(len(data))
    splits = np.array_split(shuffled_indices, 2)
    # The last split will be used as the held-out test set. The choice is arbitrary
    train_indices = splits[0]
    val_indices = splits[1]

    x_train = x.iloc[train_indices, :]
    y_train = y.iloc[train_indices, :]

    x_val = x.iloc[val_indices, :]
    y_val = y.iloc[val_indices, :]

    regressor = Regressor(x, nb_epoch=10000, lr=0.001, weight_decay=0.0001)
    regressor.optimiser = torch.optim.SGD(regressor.net.parameters(), lr=regressor.lr, weight_decay=regressor.weight_decay)
    regressor.fit(x_train, y_train, x_val, y_val)

    # regressor = Regressor(x, nb_epoch=10000, lr=0.0001, weight_decay=0.01)
    # regressor.optimiser = torch.optim.Adam(regressor.net.parameters(), lr=regressor.lr, weight_decay=regressor.weight_decay)
    # regressor.fit(x_train, y_train, x_val, y_val)

    # regressor = Regressor(x, nb_epoch=10000, lr=0.001, weight_decay=0.0001)
    # regressor.optimiser = torch.optim.Adadelta(regressor.net.parameters(), lr=regressor.lr, weight_decay=regressor.weight_decay)
    # regressor.fit(x_train, y_train, x_val, y_val)

    # regressor = Regressor(x, nb_epoch=10000, lr=0.001, weight_decay=0.0001)
    # regressor.optimiser = torch.optim.Adagrad(regressor.net.parameters(), lr=regressor.lr, weight_decay=regressor.weight_decay)
    # regressor.fit(x_train, y_train, x_val, y_val)


def hyperparameter_main():
    data = pd.read_csv("housing.csv") 
    regressor = RegressorHyperParameterSearch(data)


if __name__ == "__main__":
    hyperparameter_main()