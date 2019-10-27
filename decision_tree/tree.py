# -*- coding: utf-8 -*-

import numpy as np


class Tree(object):
    def __init__(self, pre_pruning=False, max_depth=6):
        self.feature = None
        self.label = None
        self.n_samples = None
        self.gain = None
        self.left = None
        self.right = None
        self.threshold = None
        self.pre_pruning = pre_pruning
        self.max_depth = max_depth
        self.depth = 0

    def build(self, features, target, criterion='gini'):
        self.n_samples = features.shape[0]

        # 全データが同一クラスの場合は終了
        if len(np.unique(target)) == 1:
            self.label = target[0]
            return

        best_gain = 0.0
        best_feature = None
        best_threshold = None

        # 分類木: サンプル中に最も多いクラス, 回帰木: サンプル中の平均
        if criterion in {'gini', 'entropy', 'error'}:
            self.label = max(target, key=lambda c: len(target[target == c]))
        else:
            self.label = np.mean(target)

        # ノードの不純度を計算
        impurity_node = self._calc_impurity(criterion, target)

        for col in range(features.shape[1]):
            feature_level = np.unique(features[:, col])
            thresholds = (feature_level[:-1] + feature_level[1:]) / 2.0

            # 情報利得の高い分岐点の探索
            for threshold in thresholds:
                target_l = target[features[:, col] <= threshold]
                impurity_l = self._calc_impurity(criterion, target_l)
                n_l = target_l.shape[0] / self.n_samples

                target_r = target[features[:, col] > threshold]
                impurity_r = self._calc_impurity(criterion, target_r)
                n_r = target_r.shape[0] / self.n_samples

                # 情報利得 (information gain): IG = node - (left + right)
                ig = impurity_node - (n_l * impurity_l + n_r * impurity_r)

                if ig > best_gain or best_threshold is None or best_feature is None:
                    best_gain = ig
                    best_feature = col
                    best_threshold = threshold

        self.feature = best_feature
        self.gain = best_gain
        self.threshold = best_threshold
        if self.pre_pruning is False or self.depth < self.max_depth:
            self._divide_tree(features, target, criterion)
        else:
            self.feature = None

    def _divide_tree(self, features, target, criterion):
        features_l = features[features[:, self.feature] <= self.threshold]
        target_l = target[features[:, self.feature] <= self.threshold]
        self.left = Tree(self.pre_pruning, self.max_depth)
        self.left.depth = self.depth + 1
        self.left.build(features_l, target_l, criterion)

        features_r = features[features[:, self.feature] > self.threshold]
        target_r = target[features[:, self.feature] > self.threshold]
        self.right = Tree(self.pre_pruning, self.max_depth)
        self.right.depth = self.depth + 1
        self.right.build(features_r, target_r, criterion)

    def _calc_impurity(self, criterion, target):
        classes = np.unique(target)

        if criterion == 'gini':
            return self._gini(target, classes)
        elif criterion == 'entropy':
            return self._entropy(target, classes)
        elif criterion == 'error':
            return self._classification_error(target, classes)
        elif criterion == 'mse':
            return self._mse(target)
        else:
            return self._gini(target, classes)

    def _gini(self, target, classes):
        n = target.shape[0]
        return 1.0 - sum([(len(target[target == c]) / n) ** 2 for c in classes])

    def _entropy(self, target, classes):
        n = target.shape[0]
        entropy = 0.0
        for c in classes:
            p = len(target[target == c]) / n
            if p > 0.0:
                entropy -= p * np.log2(p)
        return entropy

    def _classification_error(self, target, classes):
        n = target.shape[0]
        return 1.0 - max([len(target[target == c]) / n for c in classes])

    def _mse(self, target):
        y_hat = np.mean(target)
        return np.square(target - y_hat).mean()

    # 決定木の事後剪定
    def prune(self, method, max_depth, min_criterion, n_samples):
        if self.feature is None:
            return

        self.left.prune(method, max_depth, min_criterion, n_samples)
        self.right.prune(method, max_depth, min_criterion, n_samples)

        pruning = False

        # 剪定判定
        if method == 'impurity' and self.left.feature is None and self.right.feature is None:
            if (self.gain * self.n_samples / n_samples) < min_criterion:
                pruning = True
        elif method == 'depth' and self.depth >= max_depth:
            pruning = True

        # 剪定により過学習を抑制
        if pruning is True:
            self.left = None
            self.right = None
            self.feature = None

    def predict(self, d):
        if self.feature is None:
            return self.label
        else:
            if d[self.feature] <= self.threshold:
                return self.left.predict(d)
            else:
                return self.right.predict(d)

    def show_tree(self, depth, cond):
        base = '    ' * depth + cond
        # Leaf
        if self.feature is None:
            print(base + '{value: ' + str(self.label) + ', samples: ' + str(self.n_samples) + '}')
        else:
            # Node
            print(base + 'if X[' + str(self.feature) + '] <= ' + str(self.threshold))
            self.left.show_tree(depth+1, 'then ')
            self.right.show_tree(depth+1, 'else ')
