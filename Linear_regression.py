import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive.")
        if n_iterations <= 0:
            raise ValueError("Number of iterations must be positive.")
        
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    
    def MSE(self, y, y_pred):
        """Calculate Mean Squared Error."""
        return np.mean((y - y_pred) ** 2)
    
    def MAE(self, y, y_pred):
        """Calculate Mean Absolute Error."""
        return np.mean(np.abs(y - y_pred))
    
    def gradient_descent(self, X, y):
        """Perform gradient descent to learn weights."""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        
        return self.weights, self.bias
    
    def fit(self, X, y):
        """Fit the model to the data."""
        self.gradient_descent(X, y)

    def predict(self, X):
        """Make predictions using the learned weights."""
        return np.dot(X, self.weights) + self.bias
    

# Sample data
height = np.array([1.47, 1.50, 1.52, 1.55, 1.57, 1.60, 1.63, 1.65, 1.68, 1.70, 1.73, 1.75, 1.78, 1.80, 1.83])
weight = np.array([52.21, 53.12, 54.48, 55.84, 57.20, 58.57, 59.93, 61.29, 63.11, 64.47, 66.28, 68.10, 69.92, 72.19, 74.46])

X = height.reshape(-1, 1)
y = weight

# Create and fit the model
model = LinearRegression(learning_rate=0.001, n_iterations=1000)
model.fit(X, y)
predictions = model.predict(X)
w, b = model.weights, model.bias
# Print performance metrics
print("Mean Squared Error:", model.MSE(y, predictions))
print("Mean Absolute Error:", model.MAE(y, predictions))
print("Weights:", w)
print("Bias:", b)

# Plotting
plt.scatter(height, weight, color='blue', label='Data points')
plt.plot(X, predictions, color='red', label='Regression line')
plt.xlabel('Height (m)')
plt.ylabel('Weight (kg)')
plt.title('Height vs Weight Linear Regression')
plt.legend()
plt.show()