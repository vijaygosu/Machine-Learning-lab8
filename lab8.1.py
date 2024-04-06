import numpy as np
from collections import Counter

def calculate_entropy(y):
    class_counts = np.bincount(y)
    probabilities = class_counts / len(y)
    entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
    return entropy

def calculate_information_gain(X, y, feature_index):
    total_entropy = calculate_entropy(y)
    
    values, counts = np.unique(X[:, feature_index], return_counts=True)
    
    # Calculate weighted entropy for each value of the feature
    weighted_entropy = np.sum([(counts[i] / len(X)) * calculate_entropy(y[X[:, feature_index] == values[i]]) for i in range(len(values))])
    
    # Calculate information gain
    information_gain = total_entropy - weighted_entropy
    return information_gain
def find_root_node_value(X, y):
    max_information_gain = -1
    root_feature_value = None
    
    # Iterate over each feature to find the one with maximum information gain
    for i in range(X.shape[1]):
        information_gain = calculate_information_gain(X, y, i)
        if information_gain > max_information_gain:
            max_information_gain = information_gain
            root_feature_value = X[0, i]  # Set root feature value to the value of the first instance
    
    return root_feature_value


# Example usage:
# Define your dataset
X = np.array([
    [8, 'A', 'X'],
    [7, 'C', 'Y'],
    [9, 'A', 'Y'],
    [4, 'B', 'X'],
    [5, 'B', 'X']
])
y = np.array([2, 7, 0, 6, 3])

# Find the root node
root_feature_index = find_root_node_value(X, y)
print("Root feature :", root_feature_index)
