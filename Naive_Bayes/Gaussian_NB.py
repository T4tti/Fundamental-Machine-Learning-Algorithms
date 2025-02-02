import numpy as np

# Steps
# Train the model
# 1. Calculate the mean and variance for each class value and input value

# Predict the model
# 1. Calculate the Gaussian probability distribution function for each class value and input value
# 2. Multiply the probabilities for each input value
# 3. Classify the input value based on the highest probability

class Gaussian_NB:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # Calculate mean, variance, and prior for each class
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)
        
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        posteriors = []

        # Calculate posterior probability for each class
        for idx, c in enumerate(self.classes):
            prior = np.log(self._priors[idx])
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)
        
        # Return class with the highest posterior probability
        return self.classes[np.argmax(posteriors)]
    
    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
    
# Example
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

# Load dataset
data = datasets.load_iris() 
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Fit model
clf = Gaussian_NB()
clf.fit(X_train, y_train)

# Predict
predictions = clf.predict(X_test)
print(predictions)  

# Evaluate
accuracy = np.mean(predictions == y_test)
print(f'Accuracy: {accuracy}')

# Visualize
# Biểu đồ 1: Actual
plt.subplot(1, 2, 1)  # 1 hàng, 2 cột, vị trí 1
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Actual')

# Biểu đồ 2: Predicted
plt.subplot(1, 2, 2)  # 1 hàng, 2 cột, vị trí 2
plt.scatter(X_test[:, 0], X_test[:, 1], c=predictions, cmap='viridis')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Predicted')

# Hiển thị figure
plt.tight_layout()  # Tự động điều chỉnh khoảng cách giữa các biểu đồ
plt.show()



