import numpy as np

class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        # model parameters
        self.theta = None

        # iterations history
        self.Js = []
        self.thetas = []

    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        The learned parameters must be saved in self.theta.
        This function has no return value.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """
        # set random seed
        np.random.seed(self.random_state)
        X = self.apply_bias_trick(X)
        self.theta = np.random.random(X.shape[1])
        
        for i in range(self.n_iter):
              
          sigmoid = self.sigmoid(X.dot(self.theta))
          gradient = self.eta * (X.T.dot(sigmoid - y))
          self.theta = self.theta - gradient
          self.thetas.append(self.theta)
          self.Js.append(self.cost_function(sigmoid, y))
          if i > 1 and (self.Js[-2] - self.Js[-1]) < self.eps:
            break
        # print(self.Js)

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        X = self.apply_bias_trick(X)
        h_x = self.sigmoid(X.dot(self.theta))
        preds = np.where(h_x > 0.5, 1,0)
        return preds
    
    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))
    
    def cost_function(self, h, y):
        m = h.shape[0]

        #cost = -np.mean(y * np.log(h) + (1-y) * np.log(1-h))
        return (y.dot(np.log(h)) + (1-y).dot(np.log(1-h))) / -m

    
    def apply_bias_trick(self, X):
      """
      Applies the bias trick to the input data.

      Input:
      - X: Input data (m instances over n features).

      Returns:
      - X: Input data with an additional column of ones in the
          zeroth position (m instances over n+1 features).
      """
      ones_matrix = np.ones(X.shape[0]) # create a vector of ones
      return np.column_stack((ones_matrix, X))
    

def cross_validation(X, y, folds, algo, random_state):
    """
    This function performs cross validation as seen in class.

    1. shuffle the data and creates folds
    2. train the model on each fold
    3. calculate aggregated metrics

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    folds : number of folds (int)
    algo : an object of the classification algorithm
    random_state : int
      Random number generator seed for random weight
      initialization.

    Returns the cross validation accuracy.
    """

    cv_accuracy = 0

    # set random seed
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(X.shape[0])
    X_shuffled = X[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    num_of_elements = X.shape[0] // folds
    elements_lst = list(range(X.shape[0]))
    for i in range(1, folds + 1):
      start_threshhold = (i - 1) * num_of_elements
      end_threshhold = i * num_of_elements
      test_list = elements_lst[start_threshhold:end_threshhold]
      X_train = np.delete(X_shuffled, test_list, axis = 0)
      X_test = X_shuffled[test_list]
      y_train = np.delete(y_shuffled, test_list, axis = 0)
      y_test = y_shuffled[test_list]
      algo.fit(X_train, y_train)
      test_predict = algo.predict(X_test)
      fold_accuracy = np.mean(test_predict == y_test)
      cv_accuracy += fold_accuracy
      
    return cv_accuracy / folds

def norm_pdf(data, mu, sigma):
    """
    Calculate normal desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mu and sigma for the given x.    
    """
    p = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(data - mu)**2 /(2 * (sigma**2)))
    return p

class EM(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        np.random.seed(self.random_state)

        self.responsibilities = None
        self.weights = None
        self.mus = None
        self.sigmas = None
        self.costs = []

    # initial guesses for parameters
    def init_params(self, data):
        """
        Initialize distribution params
        """
        self.weights = np.ones(self.k) / self.k
        self.mus = np.random.random(self.k)
        self.sigmas = np.full(self.k, np.std(data))

    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """        
        self.responsibilities =  np.zeros((data.shape[0],0))
        for i in range(self.k):
            likelihood_col_i = self.weights[i] * norm_pdf(data, self.mus[i], self.sigmas[i])
            self.responsibilities = np.c_[self.responsibilities, likelihood_col_i]

        row_sums = self.responsibilities.sum(axis=1)
        self.responsibilities = self.responsibilities / row_sums[:, np.newaxis]

    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        """
        self.weights = np.mean(self.responsibilities, axis = 0)
        self.mus = (1 / (data.shape[0] * self.weights)) * (self.responsibilities.T.dot(data).flatten())
        for i in range(self.k):
            self.sigmas[i] = self.responsibilities.T[i].dot(data - self.mus[i])
        self.sigmas = (1 / (data.shape[0] * self.weights)) * self.sigmas

    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization function in order to find params
        for the distribution.
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """        
        self.init_params(data)
        for i in range(self.n_iter):  
          if i>1 and (self.costs[-2] - self.costs[-1]) < self.eps:
              break 
          self.expectation(data)
          self.maximization(data)
          cost = self.compute_cost(data)
          print(cost)
          self.costs.append(cost)
          # for d in range(data.shape[0]):
          #     cost -= np.log2(sum(self.weights[j] * norm_pdf(data[d], self.mus[j], self.sigmas[j]) for j in range(self.k)))
          #     pass

    def compute_cost(self, data):
      cost = 0
      for x in data:
          likelihood = sum(self.weights[k] * norm_pdf(x, self.mus[k], self.sigmas[k]) for k in range(self.k))
          cost += -np.log2(likelihood)
          
      return cost

    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas

def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate gmm desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - data: A value we want to compute the distribution for.
    - weights: The weights for the GMM
    - mus: The mean values of the GMM.
    - sigmas:  The standard deviation of the GMM.
 
    Returns the GMM distribution pdf according to the given mus, sigmas and weights
    for the given data.    
    """
    pdf = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pdf

class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state=1991):
        self.k = k
        self.random_state = random_state
        self.prior = None

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return preds

def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    ''' 
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    ''' 

    lor_train_acc = None
    lor_test_acc = None
    bayes_train_acc = None
    bayes_test_acc = None

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return {'lor_train_acc': lor_train_acc,
            'lor_test_acc': lor_test_acc,
            'bayes_train_acc': bayes_train_acc,
            'bayes_test_acc': bayes_test_acc}

def generate_datasets():
    from scipy.stats import multivariate_normal
    '''
    This function should have no input.
    It should generate the two dataset as described in the jupyter notebook,
    and return them according to the provided return dict.
    '''
    dataset_a_features = None
    dataset_a_labels = None
    dataset_b_features = None
    dataset_b_labels = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return{'dataset_a_features': dataset_a_features,
           'dataset_a_labels': dataset_a_labels,
           'dataset_b_features': dataset_b_features,
           'dataset_b_labels': dataset_b_labels
           }