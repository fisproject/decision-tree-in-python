# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.join('./decision-tree/'))

import decision_tree as dt
import numpy as np


def main():
    # Create a random dataset
    rng = np.random.RandomState(1)
    X = np.sort(5 * rng.rand(80, 1), axis=0)
    y = np.sin(X).ravel()
    y[::5] += 3 * (0.5 - rng.rand(16))

    # Fit regression model
    tree = dt.DecisionTreeRegressor(
                criterion='mse',
                pre_pruning=False,
                pruning_method='depth',
                max_depth=2
           )
    tree.fit(X, y)
    tree.show_tree()

    pred = tree.predict(np.sort(5 * rng.rand(1, 1), axis=0))
    print(pred)


if __name__ == '__main__':
    main()
