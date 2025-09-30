import numpy as np

def svm_decision_function(X, w, b):
    """
    Compute SVM decision scores for input X.
    
    Args:
      X: np.ndarray, shape = [n_samples, n_features]
         Input data points.
      w: np.ndarray, shape = [n_features]
         Learned SVM weights.
      b: float
         Learned bias term.
    
    Returns:
      scores: np.ndarray, shape = [n_samples]
         Decision values for each sample (before thresholding).
    """
    # Linear SVM decision: score = X @ w + b
    scores = np.dot(X, w) + b
    return scores

# Example usage:
X = np.array([[2.5, 3.0], [1.0, -1.0], [-2.0, 2.7]])  # 3 samples, 2 features
w = np.array([0.8, -0.5])                             # weights
b = 1.2                                               # bias

decision = svm_decision_function(X, w, b)
print("SVM decision scores:", decision)
print("Final decision: ",np.sign(decision))