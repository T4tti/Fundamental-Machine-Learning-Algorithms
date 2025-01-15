import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LogisticRegression: 
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    # Phương pháp Gradient Descent với log-loss
    def gradient_descent_log_loss(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        plt.figure(figsize=(8, 6))
        plt.scatter(X, y, color='green', label="Data Points")

        for i in range(self.n_iterations):
            z = np.dot(X, self.weights) + self.bias
            y_pred = sigmoid(z)

            # Tính gradient
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Cập nhật tham số
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Vẽ đường quyết định mỗi 100 vòng lặp
            if i % 100 == 0:

                loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
                print(f"Iteration {i}: Log-Loss = {loss}")
                x_plot = np.linspace(-3, 3, 300)
                y_plot = sigmoid(self.weights[0] * x_plot + self.bias)
                plt.plot(x_plot, y_plot, label=f"Iteration {i}", alpha=0.4)

        x_plot = np.linspace(-3, 3, 300)
        y_plot = sigmoid(self.weights[0] * x_plot + self.bias)
        plt.plot(x_plot, y_plot, color='red', label="Final Decision Boundary")

        plt.title("Gradient Descent with Log-Loss")
        plt.xlabel("Feature1 (Normalized)")
        plt.ylabel("Sigmoid Output")
        plt.legend()
        plt.show()

    # Phương pháp Gradient Ascent với log-likelihood
    def gradient_ascent_log_likelihood(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        plt.figure(figsize=(8, 6))
        plt.scatter(X, y, color='green', label="Data Points")
        for i in range(self.n_iterations):
            z = np.dot(X, self.weights) + self.bias
            y_pred = sigmoid(z)

            # Tính gradient
            dw = (1 / n_samples) * np.dot(X.T, (y - y_pred))
            db = (1 / n_samples) * np.sum(y - y_pred)

            # Cập nhật tham số
            self.weights += self.learning_rate * dw
            self.bias += self.learning_rate * db

            # Vẽ đường quyết định mỗi 100 vòng lặp
            if i % 100 == 0:
                Loss = np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
                print(f"Iteration {i}: Log-Likelihood = {Loss}")
                x_plot = np.linspace(-3, 3, 300)
                y_plot = sigmoid(self.weights[0] * x_plot + self.bias)
                plt.plot(x_plot, y_plot, label=f"Iteration {i}", alpha=0.4)
                

        x_plot = np.linspace(-3, 3, 300)
        y_plot = sigmoid(self.weights[0] * x_plot + self.bias)
        plt.plot(x_plot, y_plot, color='blue', label="Final Decision Boundary")

        plt.title("Gradient Ascent with Log-Likelihood")
        plt.xlabel("Feature1 (Normalized)")
        plt.ylabel("Sigmoid Output")
        plt.legend()
        plt.show()


    def fit(self, X, y, method="gradient_descent"):
        if method == "gradient_descent_log_loss":
            self.gradient_descent_log_loss(X, y)
        elif method == "gradient_ascent_log_likelihood":
            self.gradient_ascent_log_likelihood(X, y)

    def predict_probabilities(self, X):
        z = np.dot(X, self.weights) + self.bias
        return sigmoid(z)

    def predict(self, X):
        probabilities = self.predict_probabilities(X)
        return (probabilities >= 0.5).astype(int)

# Tạo dữ liệu mẫu
data = {
    'Feature1': [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 
              2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50],
    'Label': [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)
X = df[['Feature1']].values
y = df['Label'].values

# Chuẩn hóa dữ liệu
X = (X - X.mean()) / X.std()

# Huấn luyện mô hình với Gradient Descent (log-loss)
model = LogisticRegression(learning_rate=0.1, n_iterations=600)
print("Gradient Descent with Log-Loss:")
model.fit(X, y, method="gradient_descent_log_loss")
y_pred = model.predict(X)
# Hiển thị kết quả
print("\nWeights:", model.weights)            
print("Bias:", model.bias)  
print("Predictions:", y_pred)

# Huấn luyện mô hình với Gradient Ascent (log-likelihood)
model = LogisticRegression(learning_rate=0.1, n_iterations=600)
print("\nGradient Ascent with Log-Likelihood:")
model.fit(X, y, method="gradient_ascent_log_likelihood")

# Dự đoán
y_pred = model.predict(X)

# Hiển thị kết quả
print("\nWeights:", model.weights)            
print("Bias:", model.bias)  
print("Predictions:", y_pred)
