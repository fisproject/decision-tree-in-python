# -*- coding: utf-8 -*-

import numpy as np
from .tree import Tree


class DecisionTreeClassifier(object):
    def __init__(
        self,
        criterion: str = "gini",
        pre_pruning: bool = False,
        pruning_method: str = "depth",
        max_depth: int = 3,
        min_criterion: float = 0.05,
    ):
        self.root: Tree
        self.criterion = criterion
        self.pre_pruning = pre_pruning
        self.pruning_method = pruning_method
        self.max_depth = max_depth
        self.min_criterion = min_criterion

    def fit(self, features: np.ndarray, target: np.ndarray) -> None:
        self.root = Tree(self.pre_pruning, self.max_depth)
        self.root.build(features, target, self.criterion)
        # post-pruning
        if self.pre_pruning is False:
            self.root.prune(
                self.pruning_method,
                self.max_depth,
                self.min_criterion,
                self.root.n_samples,
            )

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.array([self.root.predict(f) for f in features])

    def show_tree(self) -> None:
        self.root.show_tree(0)
