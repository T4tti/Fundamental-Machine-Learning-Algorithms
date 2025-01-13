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

    def gradient_descent(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        plt.figure(figsize=(8, 6))
        plt.scatter(X, y, color='green', label='Data points')          # Plot the data points

        for i in range(self.n_iterations):
            z = np.dot(X, self.weights) + self.bias
            y_pred = sigmoid(z)
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Vẽ sigmoid và quá trình thay đổi qua mỗi 50 vòng lặp
            if i % 50 == 0:
                x_plot = np.linspace(-3, 3, 300)
                y_plot = sigmoid(self.weights[0] * x_plot + self.bias)
                plt.plot(x_plot, y_plot, alpha=0.4)
                plt.pause(0.5)
        
        x_values = np.linspace(-3, 3, 300)
        y_values = sigmoid(self.weights[0] * x_plot + self.bias)
        plt.plot(x_values, y_values, color='red', label="Final Decision Boundary")  

        plt.title("Final Decision Boundary")
        plt.xlabel("Feature1 (Normalized)")
        plt.ylabel("Sigmoid Output")
        plt.legend()
        plt.show()

    def fit(self, X, y):
        self.gradient_descent(X, y)

    def predict_probabilities(self, X):
        z = np.dot(X, self.weights) + self.bias
        return sigmoid(z)

    def predict(self, X):
        probabilities = self.predict_probabilities(X)
        return (probabilities >= 0.5).astype(int)

# Tạo dữ liệu mẫu
data = {
    'Feature1': [2.5, 3.2, 1.7, 3.8, 2.2, 2.9, 3.6, 1.5, 2.7, 3.4],
    'Label': [0, 1, 0, 1, 0, 1, 1, 0, 0, 1]
}

df = pd.DataFrame(data)
X = df[['Feature1']].values
y = df['Label'].values

# Chuẩn hóa dữ liệu
X = (X - X.mean()) / X.std()

# Huấn luyện mô hình
model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
model.fit(X, y)

# Dự đoán
y_pred = model.predict(X)

# Hiển thị kết quả
print("Predictions:", y_pred)       