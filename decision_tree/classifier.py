#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from .tree import Tree

class DecisionTreeClassifier:
    def __init__(self, criterion='gini', prune='depth', max_depth=3, min_criterion=0.05):
        self.root = None
        self.criterion = criterion
        self.prune = prune
        self.max_depth = max_depth
        self.min_criterion = min_criterion

    def fit(self, features, target):
        self.root = Tree()
        self.root.build(features, target, self.criterion)
        self.root.prune(self.prune, self.max_depth, self.min_criterion, self.root.n_samples)

    def predict(self, features):
        return np.array([self.root.predict(f) for f in features])

    def show_tree(self):
        self.root.show_tree(0, ' ')
