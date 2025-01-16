import numpy as np
import matplotlib.pyplot as plt

def minkowski_distance(x1, x2, p): 
    """
    Hàm tính khoảng cách Minkowski giữa hai điểm.
    p=1: khoảng cách Manhattan, p=2: khoảng cách Euclid.
    """
    return np.sum(np.abs(np.array(x1) - np.array(x2)) ** p) ** (1 / p)

class KNN:
    def __init__(self, k=5, p=2):
        """
        k: số lượng hàng xóm gần nhất.
        p: bậc của khoảng cách Minkowski (p=2 là Euclid).
        """
        self.k = k
        self.p = p
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Lưu trữ tập dữ liệu huấn luyện.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        """
        Dự đoán giá trị đầu ra cho tập dữ liệu kiểm tra.
        """
        predictions = []
        for new_point in X_test:
            distances = []
            for i in range(len(self.X_train)):
                # Tính khoảng cách từ điểm kiểm tra đến các điểm huấn luyện
                distance = minkowski_distance(new_point, self.X_train[i], self.p)
                distances.append((distance, self.y_train[i]))
            
            # Lấy k điểm gần nhất
            k_nearest_neighbors = sorted(distances, key=lambda x: x[0])[:self.k]
            
            # Tính giá trị trung bình của các nhãn (y) của k điểm gần nhất
            k_nearest_values = [neighbor[1] for neighbor in k_nearest_neighbors]
            prediction = np.mean(k_nearest_values)
            predictions.append(prediction)
        
        return np.array(predictions)

# Tạo dữ liệu huấn luyện
X_train = np.array([
    [1, 1],
    [2, 1],
    [3, 2],
    [4, 3],
    [5, 3],
    [6, 4],
    [7, 5],
    [8, 6]
])

y_train = np.array([1.2, 2.3, 2.8, 3.6, 4.0, 4.5, 5.8, 6.3])

# Dữ liệu kiểm tra
X_test = np.array([
    [2, 2],
    [5, 5],
    [7, 4]
])

# Số hàng xóm
model = KNN(k=4, p=2)  # Sử dụng khoảng cách Euclid
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("Dự đoán:", predictions)



# Vẽ tập dữ liệu huấn luyện
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="viridis", label="Train Points", s=100)
plt.colorbar(label="y_train")

# Vẽ các điểm kiểm tra
plt.scatter(X_test[:, 0], X_test[:, 1], color="red", label="Test Points", s=100)

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Tập dữ liệu K-NN (hồi quy)")
plt.legend()
plt.grid()
plt.show()

