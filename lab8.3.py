import numpy as np
from collections import Counter

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, binning_type='equal_width', num_bins=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.binning_type = binning_type
        self.num_bins = num_bins
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        # Stopping criteria
        if depth == self.max_depth or len(X) < self.min_samples_split or len(np.unique(y)) == 1:
            return Counter(y).most_common(1)[0][0]

        # Find the best split
        root_feature_index = self._find_root_node(X, y)
        
        if root_feature_index is None:
            return Counter(y).most_common(1)[0][0]
        
        # Binning continuous feature if necessary
        if np.issubdtype(X[:, root_feature_index].dtype, np.number):
            X[:, root_feature_index] = self._bin_continuous_feature(X[:, root_feature_index])

        # Split the data
        values = np.unique(X[:, root_feature_index])
        sub_trees = {}
        for value in values:
            mask = X[:, root_feature_index] == value
            X_subset, y_subset = X[mask], y[mask]
            sub_trees[value] = self._build_tree(X_subset, y_subset, depth + 1)

        return {'feature_index': root_feature_index, 'sub_trees': sub_trees}

    def _calculate_entropy(self, y):
        class_counts = np.bincount(y)
        probabilities = class_counts / len(y)
        entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
        return entropy

    def _calculate_information_gain(self, X, y, feature_index):
        total_entropy = self._calculate_entropy(y)
        values, counts = np.unique(X[:, feature_index], return_counts=True)
        weighted_entropy = np.sum([(counts[i] / len(X)) * self._calculate_entropy(y[X[:, feature_index] == values[i]]) for i in range(len(values))])
        information_gain = total_entropy - weighted_entropy
        return information_gain

    def _find_root_node(self, X, y):
        max_information_gain = -1
        root_feature_index = None
        for i in range(X.shape[1]):
            information_gain = self._calculate_information_gain(X, y, i)
            if information_gain > max_information_gain:
                max_information_gain = information_gain
                root_feature_index = i
        return root_feature_index

    def _equal_width_binning(self, data):
        min_value = np.min(data.astype(float))
        max_value = np.max(data.astype(float))
        bin_width = (max_value - min_value) / self.num_bins
        bins = [min_value + i * bin_width for i in range(self.num_bins + 1)]
        binned_data = np.digitize(data, bins) - 1
        return binned_data

    def _frequency_binning(self, data):
        counts, bins = np.histogram(data.astype(float), bins=self.num_bins)
        binned_data = np.digitize(data, bins[:-1])
        return binned_data

    def _bin_continuous_feature(self, data):
        if self.binning_type == 'equal_width':
            return self._equal_width_binning(data)
        elif self.binning_type == 'frequency':
            return self._frequency_binning(data)
        else:
            raise ValueError("Invalid binning type. Please choose 'equal_width' or 'frequency'.")

    def predict(self, X):
        return np.array([self._predict_tree(x, self.tree) for x in X])

    def _predict_tree(self, x, tree):
        if isinstance(tree, dict):
            feature_index = tree['feature_index']
            value = x[feature_index]
            if np.issubdtype(value.dtype, np.number):
                value = self._bin_continuous_feature(value)
            if value in tree['sub_trees']:
                return self._predict_tree(x, tree['sub_trees'][value])
        return tree

# Example usage:
X = np.array([
    [1, 3.5, 'X'],
    [2, 2.5, 'Y'],
    [3, 4.5, 'Y'],
    [4, 3.0, 'X'],
    [5, 4.0, 'X']
])
y = np.array([0, 1, 1, 0, 0])

# Initialize and fit the decision tree
tree = DecisionTree(max_depth=3, min_samples_split=2, binning_type='equal_width', num_bins=3)
tree.fit(X, y)

# Predictions
predictions = tree.predict(X)
print("Predictions:", predictions)
