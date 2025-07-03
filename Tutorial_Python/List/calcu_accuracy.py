# Bài 7: Tính Toán Độ Chính Xác (Accuracy)

'''
Cho hai danh sách có cùng độ dài: một danh sách chứa các nhãn thực tế ('y_true') và một
danh sách chứa các nhãn dự đoán từ mô hình ('y_pred'). Hãy viết một hàm để:
• So sánh từng cặp phần tử tương ứng trong hai danh sách.
• Tính toán và trả về độ chính xác (accuracy) dưới dạng một số thực trong đoạn
[0.0, 1.0].
Input:
• 'y_true': list các nhãn thực tế.
• 'y_pred': list các nhãn dự đoán.
Output: Một số thực là giá trị độ chính xác.
Ví dụ:
Input:
y_true = ["cat", "dog", "cat", "bird"]
y_pred = ["cat", "cat", "cat", "bird"]
Output: 0.75
'''

def calculate_accuracy(y_true, y_pred):
    # Bước 1: Kiểm tra nếu danh sách đầu vào rỗng
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return 0.0
    
    correct_predictions = 0

    for true_label, pred_label in zip(y_true, y_pred):
        if true_label == pred_label:
            correct_predictions += 1

    # Bước 4: Tính toán độ chính xác
    accuracy = correct_predictions / len(y_true)

    return accuracy

# ==============================
# TEST CASES
# ==============================

# Test 1: Các nhãn là chuỗi, độ chính xác 80%
y_true1 = ["cat", "dog", "cat", "bird", "dog"]
y_pred1 = ["cat", "dog", "cat", "dog", "dog"]
assert calculate_accuracy(y_true1, y_pred1) == 0.8, "Test 1 Failed"

# Test 2: Các nhãn là số, độ chính xác 100%
y_true2 = [1, 0, 1, 1, 0]
y_pred2 = [1, 0, 1, 1, 0]
assert calculate_accuracy(y_true2, y_pred2) == 1.0, "Test 2 Failed"

# Test 3: Độ chính xác 0%
y_true3 = [0, 0, 0]
y_pred3 = [1, 1, 1]
assert calculate_accuracy(y_true3, y_pred3) == 0.0, "Test 3 Failed"

# Test 4: Danh sách rỗng
assert calculate_accuracy([], []) == 0.0, "Test 4 Failed"

# Test 5: Độ chính xác 50%
y_true5 = ["A", "B"]
y_pred5 = ["A", "C"]
assert calculate_accuracy(y_true5, y_pred5) == 0.5, "Test 5 Failed"

print("All tests passed.")