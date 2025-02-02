import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split    
from sklearn import datasets
import matplotlib.pyplot as plt

class Multinomial_NB:
    def  __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_log_prior_ = None
        self.feature_log_prob_ = None
        self.classes_ = None

    def fit(self, X, y):
        """
        Fit the Multinomial Naive Bayes model according to the training data.
        
        Parameters:
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        """

        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # Initialize arrays to store log probabilities
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        # Calculate class log priors and feature log probabilities
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.class_log_prior_[idx] = np.log(X_c.shape[0] / n_samples)
            smoothed_class_count = X_c.sum(axis=0) + self.alpha
            smoothed_class_total = smoothed_class_count.sum()
            self.feature_log_prob_[idx, :] = np.log(smoothed_class_count / smoothed_class_total)

    def predict_log_proba(self, X):
        """
        Return log-probability estimates for the test data X.

        Parameters:
        X : array-like of shape (n_samples, n_features)
        Test data.

        Returns:
        log_proba : array-like of shape (n_samples, n_classes)
        Log-probability of the samples for each class in the model.
        """
        return (X @ self.feature_log_prob_.T) + self.class_log_prior_

    def predict(self, X):
        """
        Perform classification on an array of test vectors X.
        
        Parameters:
        X : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns:
        C : array-like of shape (n_samples,)
            Predicted target values for X.
        """
        log_proba = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_proba, axis=1)]
    
# Example
# Load dataset
data = datasets.load_iris()
X, y = data.data, data.target   

# Split data into training and test sets    

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Train the model
model = Multinomial_NB()
model.fit(X_train, y_train) 

# Predict the model 
y_pred = model.predict(X_test)
print("Predictions:", y_pred)

# Evaluate the model
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)

# Visualize the data
# Biểu đồ 1: Actual
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', s=100, alpha=0.8)
plt.colorbar(label="y_train")
plt.title("Actual")  
   

# Biểu đồ 2: Predicted      
plt.subplot(1, 2, 2)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis', s=100, alpha=0.8)
plt.colorbar(label="y_pred")    
plt.title("Predicted")  
plt.show()  
