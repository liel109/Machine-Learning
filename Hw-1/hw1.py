###### Your ID ######
# ID1: 209207380
# ID2: 206910333
#####################

# imports 
import numpy as np
import pandas as pd

def preprocess(X,y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """
    mean_vec = np.mean(X,0)
    min_max_vec = X.max(0)-X.min(0)
    X = (X - mean_vec) / (min_max_vec)
    y=(y-np.mean(y)) / (y.max() - y.min())

    return X, y

def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """
    ones_matrix = np.ones(X.shape[0]) # create a vector of ones
    X = np.column_stack((ones_matrix, X)) # add the vector of ones to the input data
    
    return X

def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.  

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the cost associated with the current set of parameters (single number).
    """
    return np.sum(np.square(X.dot(theta) - y)) / (X.shape[0] * 2)

def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration
    for i in range(num_iters):
        J_history.append(compute_cost(X, y, theta))
        vec = (X.dot(theta) - y)
        gradient = vec.dot(X) / X.shape[0]
        theta = theta - gradient * alpha
    
    return theta, J_history

def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """
    X_t = np.transpose(X)
    pinv = np.matmul(np.linalg.inv(np.matmul(X_t, X)), X_t)
    
    return np.matmul(pinv,y)

def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than 1e-8. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration
    for i in range(num_iters):
        if i > 1 and (J_history[-2] - J_history[-1]) < 1e-8 :
            break
        vec = (X.dot(theta) - y)
        gradient = vec.dot(X) / X.shape[0]
        theta = theta - (gradient * alpha)
        J_history.append(compute_cost(X, y, theta))
    
    return theta, J_history

def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of alpha and train a model using 
    the training dataset. maintain a python dictionary with alpha as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {alpha_value : validation_loss}
    """
    
    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {} # {alpha_value: validation_loss}
    np.random.seed(42) # creates a random seed
    theta_guess = np.random.random(X_train.shape[1]) # creates a random vector 
    for alpha in alphas:
        theta, _ = efficient_gradient_descent(X_train, y_train, theta_guess, alpha, iterations)# train the model using the selected alpha
        alpha_dict[alpha] = compute_cost(X_val, y_val, theta)# compute the loss on the validation set

    return alpha_dict

def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_alpha: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []
    features_list = [i for i in range(X_train.shape[1])] # list of the indexes valid features to be selected
    for i in range(5):
        np.random.seed(42)
        theta_guess = np.random.random(i+2) # creates a random theta vector
        costs_dict = {}
        for feature in features_list:
            selected_features.append(feature) # add the feature to the selected features
            X_candidate = apply_bias_trick(X_train[:, selected_features])
            theta, _ = efficient_gradient_descent(X_candidate, y_train, theta_guess, best_alpha, iterations)# train the model using the selected feature
            costs_dict[feature] = compute_cost(apply_bias_trick(X_val[:, selected_features]), y_val, theta)# compute the loss on the validation set for the selected feature
            selected_features.remove(feature)# remove the feature from the selected features
        
        min_feature = min(costs_dict, key = costs_dict.get) # get the feature with the minimum loss of validation 
        selected_features.append(min_feature) # add the feature to the selected features
        features_list.remove(min_feature)# remove the feature from the list of valid features to be selected

    return selected_features

def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    df_poly = df.copy()
    for i,col1 in enumerate(df.columns):
        for col2 in df.columns[i:]:
         
            if col1 == col2:
                new_col = df[col1] ** 2
                new_col.name = f"{col1}^2"
                df_poly = pd.concat([df_poly,new_col],axis=1)
                
            else:
                new_col = df[col1] * df[col2]
                new_col.name = f"{col1}*{col2}"
                df_poly = pd.concat([df_poly,new_col],axis=1)

    return df_poly
