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
        
        plt.figure(figsize=(8, 6))  # Initialize the plot
        plt.scatter(X, y, color='blue', label='Data points')  # Plot the data points

        for _ in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Check values of weights and bias
            if abs(self.learning_rate * dw) < 1e-5 and abs(self.learning_rate * db) < 1e-5:
                break

            # Plot the current regression line
            if _ % 50 == 0:  # Update the plot every 50 iterations
                x_values = np.array([X.min(), X.max()])
                y_values = self.weights[0] * x_values + self.bias
                plt.plot(x_values, y_values, color='green', alpha=0.1)
                plt.pause(0.5)
        
        # Plot the final regression line
        x_values = np.array([X.min(), X.max()])
        y_values = self.weights[0] * x_values + self.bias
        plt.plot(x_values, y_values, color='red', label='Final Regression Line')
        
        plt.xlabel('Height (m)')
        plt.ylabel('Weight (kg)')
        plt.title('Height vs Weight Linear Regression')
        plt.legend()
        plt.show()

    
    def fit(self, X, y):
        """Fit the model to the data."""
        X = np.array(X).reshape(-1, 1)  # Ensure X is a 2D array
        self.gradient_descent(X, y)

    def predict(self, X):
        """Make predictions using the learned weights."""
        X = np.array(X).reshape(-1, 1)  # Ensure X is a 2D array
        return np.dot(X, self.weights) + self.bias
    

# Create some data
X = np.arange(-5, 5, 0.5)
n_samples = len(X)
noise = np.random.normal(0, 1, n_samples)
y = 3 * X + 2 + noise

# Create and fit the model
model = LinearRegression(learning_rate=0.001, n_iterations=1000)
model.fit(X, y)
predictions = model.predict(X)

# Print performance metrics
print("Mean Squared Error:", model.MSE(y, predictions))
print("Mean Absolute Error:", model.MAE(y, predictions))
print("Weights:", model.weights)
print("Bias:", model.bias)