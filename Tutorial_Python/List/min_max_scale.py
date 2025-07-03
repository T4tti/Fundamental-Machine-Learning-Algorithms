# Bài 6: Chuẩn Hóa Đặc Trưng (Feature Scaling)
'''
Cho một danh sách các số thực đại diện cho một đặc trưng, hãy viết một hàm thực hiện:
• Áp dụng công thức Min-Max Scaling để chuẩn hóa tất cả các giá trị trong danh sách
về đoạn [0, 1].
• Nếu tất cả các giá trị trong danh sách bằng nhau (Xmax = Xmin), hãy trả về một
danh sách chứa các giá trị 0.
Input: Một list các số (số nguyên hoặc số thực).
Output: Một list mới với các giá trị đã được chuẩn hóa theo Min-Max Scaling.
Ví dụ:
Input: [10, 20, 50]
Output: [0.0, 0.25, 1.0]
'''

def min_max_scale(data):
    # Bước 1: Kiểm tra các trường hợp đặc biệt (ví dụ: danh sách rỗng)
    if not data:
        return []
    
    # Bước 2: Tìm giá trị nhỏ nhất (min_val) và lớn nhất (max_val) trong danh sách
    min_val = min(data)
    max_val = max(data)

    # Bước 3: Kiểm tra nếu min_val bằng max_val để tránh chia cho 0
    if min_val == max_val:
        return [0 for _ in data]
    
    # Bước 4: Khởi tạo danh sách kết quả
    scaled_data = []

    # Duyệt qua từng giá trị trong danh sách đầu vào:
    for val in data:
        # Áp dụng công thức Min-Max Scaling và thêm vào danh sách kết quả
        scaled_value = (val - min_val) / (max_val - min_val)
        scaled_data.append(scaled_value)

    # Bước 5: Trả về danh sách đã được chuẩn hóa
    return scaled_data

# ==============================
# TEST CASES
# ==============================

# Test 1: Chuẩn hóa danh sách số nguyên dương
assert min_max_scale([10, 20, 50, 30]) == [0.0, 0.25, 1.0, 0.5], "Test 1 Failed"

# Test 2: Danh sách chứa số âm và số 0
assert min_max_scale([-10, 0, 10]) == [0.0, 0.5, 1.0], "Test 2 Failed"

# Test 3: Tất cả các phần tử giống nhau
assert min_max_scale([5, 5, 5, 5]) == [0.0, 0.0, 0.0, 0.0], "Test 3 Failed"

# Test 4: Danh sách đầu vào rỗng
assert min_max_scale([]) == [], "Test 4 Failed"

# Test 5: Dữ liệu đã trong khoảng [0, 1], kiểm tra sai số floating-point
scaled_data = min_max_scale([0.1, 0.5, 0.9])
expected_data = [0.0, 0.5, 1.0]
assert all(abs(a - b) < 1e-9 for a, b in zip(scaled_data, expected_data)), "Test 5 Failed"

print("All tests passed.")