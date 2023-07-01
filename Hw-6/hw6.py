import numpy as np

def get_random_centroids(X, k):
    '''
    Each centroid is a point in RGB space (color) in the image. 
    This function should uniformly pick `k` centroids from the dataset.
    Input: a single image of shape `(num_pixels, 3)` and `k`, the number of centroids. 
    Notice we are flattening the image to a two dimentional array.
    Output: Randomly chosen centroids of shape `(k,3)` as a numpy array. 
    '''
    centroids = []
    random_indices = np.random.choice(X.shape[0], size=k, replace=False)
    centroids = X[random_indices]
    return np.asarray(centroids).astype(np.float)

def calculate_lp_distance(X, centroids, p=2):
    return (np.sum((np.absolute((X-centroids)))**p, axis=1)**(1/p)).T

def lp_distance(X, centroids, p=2):
    '''
    Inputs: 
    A single image of shape (num_pixels, 3)
    The centroids (k, 3)
    The distance parameter p

    output: numpy array of shape `(k, num_pixels)` thats holds the distances of 
    all points in RGB space from all centroids
    '''
    distances = np.array([calculate_lp_distance(X, c, p) for c in centroids])
    return distances

def kmeans_calculation(X, centroids, p, max_iter):
    classes = []
    k=centroids.shape[0]
    for i in range(max_iter):
        prev_centroids = np.copy(centroids)
        classes = np.argmin(lp_distance(X,centroids, p), axis=0)
        centroids = np.array([np.mean(X[classes==j], axis=0) for j in range(k)])
        if np.array_equal(centroids, prev_centroids):
            break
    
    return centroids, classes

def kmeans(X, k, p ,max_iter=100):
    """
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    centroids = get_random_centroids(X, k)
    return kmeans_calculation(X,centroids,p,max_iter)

def kmeans_pp(X, k, p ,max_iter=100):
    """
    Your implenentation of the kmeans++ algorithm.
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    centroids = []
    X_copy = np.copy(X)
    chosen_centroid = np.random.choice(X_copy.shape[0])
    centroids = X_copy[chosen_centroid]
    X_copy = np.delete(X_copy, chosen_centroid, axis=0)

    for i in range(k-1):
        distnaces = lp_distance(X_copy,centroids, p)
        min_distances_squared = np.amin(distnaces, axis = 0)**2
        probabilities = min_distances_squared / np.sum(min_distances_squared)
        chosen_centroid = np.random.choice(X_copy.shape[0], p=probabilities)
        centroids = np.vstack((centroids, X_copy[chosen_centroid]))
        X_copy = np.delete(X_copy, chosen_centroid, axis=0)

    return kmeans_calculation(X,centroids,p,max_iter)

def inertia(X, classes, centroids, p=2):
    k = centroids.shape[0]
    return np.sum([np.sum(calculate_lp_distance(X[classes==j], centroids[j], p)**p, axis=0) for j in range(k)])
