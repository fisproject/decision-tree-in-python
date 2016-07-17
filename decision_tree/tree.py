#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np

class Tree:
    def __init__(self):
        self.feature = None
        self.label = None
        self.n_samples = None
        self.gain = None
        self.left = None
        self.right = None
        self.threshold = None
        self.depth = 0

    def build(self, features, target, criterion='gini'):
        # 全データが同一クラスの場合は終了
        if len(np.unique(target)) == 1:
            self.label = target[0]
            return

        self.n_samples = features.shape[0] # サンプルサイズ
        best_gain = 0.0
        best_feature = None
        best_threshold = None

        # サンプル中に最も多いクラスを設定
        self.label= max([(c, len(target[target==c])) for c in np.unique(target)],
            key=lambda x:x[1])[0]

        # ノードの不純度
        impurity_node = self._calc_impurity(criterion, target)

        for col in xrange(features.shape[1]):
            feature_level = np.unique(features[:,col])
            thresholds = (feature_level[:-1] + feature_level[1:]) / 2.0

            # 探索
            for threshold in thresholds:
                target_l = target[features[:,col] <= threshold]
                impurity_l = self._calc_impurity(criterion, target_l)
                n_l = float(target_l.shape[0]) / self.n_samples

                target_r = target[features[:,col] > threshold]
                impurity_r = self._calc_impurity(criterion, target_r)
                n_r = float(target_r.shape[0]) / self.n_samples

                # 情報利得 (information gain): IG = node - (left + right)
                ig = impurity_node - (n_l * impurity_l + n_r * impurity_r)

                # 情報利得の最大化
                if ig > best_gain:
                    best_gain = ig
                    best_feature = col
                    best_threshold = threshold

        self.feature = best_feature
        self.gain = best_gain
        self.threshold = best_threshold
        self._divide_tree(features,target, criterion)

    def _divide_tree(self, features, target, criterion):
        features_l = features[features[:, self.feature] <= self.threshold]
        target_l = target[features[:, self.feature] <= self.threshold]
        self.left = Tree()
        self.left.depth = self.depth + 1
        self.left.build(features_l, target_l, criterion)

        features_r = features[features[:, self.feature] > self.threshold]
        target_r = target[features[:, self.feature] > self.threshold]
        self.right = Tree()
        self.right.depth = self.depth + 1
        self.right.build(features_r, target_r, criterion)


    def _calc_impurity(self, criterion, target):
        c = np.unique(target) # クラス数
        s = target.shape[0] # サンプルサイズ

        if criterion == 'gini':
            return self._gini(target, c, s)
        elif criterion == 'entropy':
            return self._entropy(target, c, s)
        elif criterion == 'error':
            return self._error(target, c, s)
        else:
            return self._gini(target, c, s)

    # ジニ不純度
    def _gini(self, target, n_classes, n_samples):
        gini_index = 1.0
        gini_index -= sum([(len(target[target==c]) / n_samples) ** 2.0 for c in n_classes])
        return gini_index

    # エントロピー
    def _entropy(self, target, n_classes, n_samples):
        entropy = 0.0

        for c in n_classes:
            p = float(len(target[target==c])) / n_samples
            if p > 0.0:
                entropy -= p * np.log2(p)
        return entropy

    # 分類誤差
    def _error(self, target, n_classes, n_samples):
        return 1.0 - max([len(target[target==c]) / n_samples for c in n_classes])

    # 決定木の剪定
    def prune(self, method, max_depth, min_criterion, n_samples):
        if self.feature == None:
            return

        self.left.prune(method, max_depth, min_criterion, n_samples)
        self.right.prune(method, max_depth, min_criterion, n_samples)

        pruning = False

        # 剪定判定
        if method == 'impurity' and self.left.feature == None and self.right.feature == None: # Leaf
            if (self.gain * float(self.n_samples) / n_samples) < min_criterion:
                pruning = True
        elif method == 'depth' and self.depth >= max_depth:
            pruning = True

        # 剪定により過学習を抑制
        if pruning is True:
            self.left = None
            self.right = None
            self.feature = None

    def predict(self, d):
        if self.feature != None: # Node
            if d[self.feature] <= self.threshold:
                return self.left.predict(d)
            else:
                return self.right.predict(d)
        else: # Leaf
            return self.label

    def show_tree(self, depth, cond):
        base = '    ' * depth + cond
        if self.feature != None: # Node
            print(base + 'if X[' + str(self.feature) + '] <= ' + str(self.threshold))
            self.left.show_tree(depth+1, 'then ')
            self.right.show_tree(depth+1, 'else ')
        else: # Leaf
            print(base + '{class: ' + str(self.label) + ', samples: ' + str(self.n_samples) + '}')
