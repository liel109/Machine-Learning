import numpy as np
import matplotlib.pyplot as plt

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5 : 0.45,
             0.25 : 1.32,
             0.1 : 2.71,
             0.05 : 3.84,
             0.0001 : 100000},
         2: {0.5 : 1.39,
             0.25 : 2.77,
             0.1 : 4.60,
             0.05 : 5.99,
             0.0001 : 100000},
         3: {0.5 : 2.37,
             0.25 : 4.11,
             0.1 : 6.25,
             0.05 : 7.82,
             0.0001 : 100000},
         4: {0.5 : 3.36,
             0.25 : 5.38,
             0.1 : 7.78,
             0.05 : 9.49,
             0.0001 : 100000},
         5: {0.5 : 4.35,
             0.25 : 6.63,
             0.1 : 9.24,
             0.05 : 11.07,
             0.0001 : 100000},
         6: {0.5 : 5.35,
             0.25 : 7.84,
             0.1 : 10.64,
             0.05 : 12.59,
             0.0001 : 100000},
         7: {0.5 : 6.35,
             0.25 : 9.04,
             0.1 : 12.01,
             0.05 : 14.07,
             0.0001 : 100000},
         8: {0.5 : 7.34,
             0.25 : 10.22,
             0.1 : 13.36,
             0.05 : 15.51,
             0.0001 : 100000},
         9: {0.5 : 8.34,
             0.25 : 11.39,
             0.1 : 14.68,
             0.05 : 16.92,
             0.0001 : 100000},
         10: {0.5 : 9.34,
              0.25 : 12.55,
              0.1 : 15.99,
              0.05 : 18.31,
              0.0001 : 100000},
         11: {0.5 : 10.34,
              0.25 : 13.7,
              0.1 : 17.27,
              0.05 : 19.68,
              0.0001 : 100000}}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns:
    - gini: The gini impurity value.
    """
    labels_dict = count_labels(data, -1)
    data_size = data.shape[0] 
    gini = 1 - sum(((num_of_elements/data_size)**2 for num_of_elements in labels_dict.values()))
    return gini

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    labels_dict = count_labels(data, -1)
    data_size = data.shape[0]
    entropy = -sum(((num_of_elements/data_size)*np.log2(num_of_elements/data_size) for num_of_elements in labels_dict.values()))
    return entropy

def goodness_of_split(data, feature, impurity_func, gain_ratio=False):
    """
    Calculate the goodness of split of a dataset given a feature and impurity function.
    Note: Python support passing a function as arguments to another function
    Input:
    - data: any dataset where the last column holds the labels.
    - feature: the feature index the split is being evaluated according to.
    - impurity_func: a function that calculates the impurity.
    - gain_ratio: goodness of split or gain ratio flag.

    Returns:
    - goodness: the goodness of split value
    - groups: a dictionary holding the data after splitting 
              according to the feature values.
    """
    goodness = 0
    groups = {} # groups[feature_value] = data_subset
    values_dict = count_labels(data, feature)
    data_size = data.shape[0]
    sum_of_impurities = 0
    split_information = 0

    for value in values_dict:
        sub_data = data[data[:,feature] == value] # get the subset of data with the current value
        sub_data_size = values_dict[value]
        groups[value] = sub_data
        sum_of_impurities += (sub_data_size / data_size) * impurity_func(sub_data)
        if gain_ratio:
            split_information += (sub_data_size / data_size) * np.log2(sub_data_size / data_size)
    
    goodness = impurity_func(data) - sum_of_impurities
    if gain_ratio and split_information != 0:
        goodness =  goodness / -split_information
    
    return goodness, groups

class DecisionNode:

    def __init__(self, data, feature=-1,depth=0, chi=1, max_depth=1000, gain_ratio=False):
        
        self.data = data # the relevant data for the node
        self.feature = feature # column index of criteria being tested
        self.pred = self.calc_node_pred() # the prediction of the node
        self.depth = depth # the current depth of the node
        self.children = [] # array that holds this nodes children
        self.children_values = []
        self.terminal = False # determines if the node is a leaf
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio 
    
    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        labels_dict = count_labels(self.data, -1)
        
        return max(labels_dict, key = labels_dict.get)
        
    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        self.children.append(node)
        self.children_values.append(val)
     
    def split(self, impurity_func):
        """
        Splits the current node according to the impurity_func. This function finds
        the best feature to split according to, and create the corresponding children.
        This function should support pruning according to chi and max_depth.

        Input:
        - The impurity function that should be used as the splitting criteria

        This function has no return value
        """
        max_goodness = 0
        max_groups = []

        for feature in range(self.data.shape[1] - 1):
            goodness_value, groups = goodness_of_split(self.data, feature, impurity_func, self.gain_ratio)
            if goodness_value > max_goodness:
                max_groups = groups
                max_goodness = goodness_value
                self.feature = feature
        
        if len(max_groups) <= 1:
            self.terminal = True
            return
         
        #create children
        for value, data in max_groups.items():
            new_node = DecisionNode(data, depth=self.depth+1, chi=self.chi, max_depth=self.max_depth, gain_ratio=self.gain_ratio)
            self.add_child(new_node, value)


    def calc_chi_value(self):
        """
        Calculate the chi value of the current node.

        Returns:
        - chi_value: the chi value of the node
        """
        
        chi_value = 0
        labels_dict = count_labels(self.data, -1)
        # calculate the probabilities of each label and store them in a dictionary
        probabilities = {label:num_of_labels / self.data.shape[0] for label,num_of_labels in labels_dict.items()}
        for child in self.children:
            child_labels_dict = count_labels(child.data, -1)
            Df = child.data.shape[0]
            for label in probabilities:
                E = Df * probabilities[label]
                if label in child_labels_dict:
                    num_of_instances = child_labels_dict[label]
                else:
                    num_of_instances = 0
                chi_value += (num_of_instances - E)**2 / E

        return chi_value

    def is_random(self):
        """
        Decides whether the current node split is random.

        Returns:
        - boolean: if the split was random
        """
        DOF = len(self.children_values) - 1 #degrees of freedom

        if self.chi == 1 or DOF < 1: 
            return False

        return self.calc_chi_value() < chi_table[DOF][self.chi]


def build_tree(data, impurity, gain_ratio=False, chi=1, max_depth=1000):
    """
    Build a tree using the given impurity measure and training dataset. 
    You are required to fully grow the tree until all leaves are pure unless
    you are using pruning

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.
    - gain_ratio: goodness of split or gain ratio flag

    Output: the root node of the tree.
    """
    root = DecisionNode(data,chi = chi, max_depth=max_depth, gain_ratio=gain_ratio)
    queue = [root]
    while len(queue) > 0:
        node = queue.pop(0)
        labels_dict = count_labels(node.data, -1)
        if len(labels_dict) == 1 or node.depth == max_depth:
            node.terminal = True
        else:
            node.split(impurity)
            if node.is_random():
                node.terminal = True
                continue
            queue += node.children

    return root

def predict(root, instance):
    """
    Predict a given instance using the decision tree
 
    Input:
    - root: the root of the decision tree.
    - instance: an row vector from the dataset. Note that the last element 
                of this vector is the label of the instance.
 
    Output: the prediction of the instance.
    """
    pointer = root
    while not pointer.terminal:
        attr = instance[pointer.feature]
        try:
            pointer = pointer.children[pointer.children_values.index(attr)]
        except:
            break
        
    return pointer.pred

def calc_accuracy(node, dataset):
    """
    Predict a given dataset using the decision tree and calculate the accuracy
 
    Input:
    - node: a node in the decision tree.
    - dataset: the dataset on which the accuracy is evaluated
 
    Output: the accuracy of the decision tree on the given dataset (%).
    """
    accuracy = 0
    for row in dataset:
        if predict(node, row) == row[-1]:
            accuracy += 1

    return (accuracy / dataset.shape[0]) * 100


def depth_pruning(X_train, X_test):
    """
    Calculate the training and testing accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously.

    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels
 
    Output: the training and testing accuracies per max depth
    """
    training = []
    testing  = []
    for max_depth in range(1,11):
        root = build_tree(X_train, calc_entropy, True, max_depth=max_depth)
        training.append(calc_accuracy(root, X_train))
        testing.append(calc_accuracy(root, X_test))
    
    return training, testing


def chi_pruning(X_train, X_test):

    """
    Calculate the training and testing accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels
 
    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_testing_acc: the testing accuracy per chi value
    - depths: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_testing_acc  = []
    depth = []
    for chi_val in [1, 0.5, 0.25, 0.1, 0.05, 0.0001]:
        root = build_tree(X_train, calc_entropy, True, chi=chi_val)
        chi_training_acc.append(calc_accuracy(root, X_train))
        chi_testing_acc.append(calc_accuracy(root, X_test))
        depth.append(get_tree_depth(root))

    return chi_training_acc, chi_testing_acc, depth


def count_nodes(node, nodes_counter = 0):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of nodes in the tree.
    """
    if node.terminal:
        return 1
    else:
        count = 1
        for child in node.children:
            count += count_nodes(child)
        return count


def count_labels(data, feature):
    """Creates a dictionary of the labels and their count in a certain column in the data

    Args:
        data: the data to count the labels in
        feature: the index of the column

    Returns:
        dict: a dictionary of the labels and their count
    """
    arr, count = np.unique(data[:, feature], return_counts= True)
    return dict(zip(arr,count))


def get_tree_depth(root):
    """Calculates the depth of a tree

    Args:
        root: the root of the tree

    Returns:
        Int: the depth of the tree
    """
    if root.terminal:
        return root.depth
    
    return max([get_tree_depth(child) for child in root.children])






