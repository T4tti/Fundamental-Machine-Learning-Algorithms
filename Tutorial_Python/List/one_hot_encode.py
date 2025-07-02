# Bài 4: Xử Lý Nhãn Văn Bản Bằng Kỹ Thuật One-Hot
'''
Cho một danh sách các nhãn văn bản và danh sách tất cả các lớp có thể có (đã được sắp
xếp), hãy viết một hàm thực hiện:
• Chuyển đổi từng nhãn thành một vector one-hot có độ dài bằng số lớp.
• Trong mỗi vector, chỉ có một phần tử là 1 ở vị trí tương ứng với lớp của nhãn đó,
các phần tử còn lại là 0.
    Input:
        • labels: list các nhãn văn bản (ví dụ: ["cat", "dog"])
        • classes: list tất cả các lớp có thể có, đã được sắp xếp thứ tự (ví dụ: ["bird",
        "cat", "dog"])
    Output: List các vector one-hot tương ứng với từng nhãn đầu vào.
Ví dụ:
Input:
labels = ["cat", "dog"]
classes = ["bird", "cat", "dog"]
Output:
[[0, 1, 0],
[0, 0, 1]]
'''
def one_hot_encode(labels, classes):
    # Bước 1: Xác định số lượng lớp cần encode (số chiều của vector one-hot)
    num_classes = len(classes)

    # Bước 2: Khởi tạo danh sách kết quả rỗng
    one_hot_vectors = []

    # Bước 3: Duyệt qua từng nhãn trong danh sách labels
    for label in labels:
        # Tìm chỉ số của nhãn trong danh sách classes
        if label in classes:
            index = classes.index(label)

        # Tạo vector one-hot với tất cả phần tử là 0, chỉ có vị trí tương ứng là 1
        vector = [0] * num_classes
        vector[index] = 1

        # Thêm vector này vào danh sách kết quả
        one_hot_vectors.append(vector)
    # Bước 4: Trả về danh sách các vector one-hot
    return one_hot_vectors

# Test 1: Danh sách nhãn đầy đủ, kiểm tra thứ tự và ánh xạ chính xác
assert one_hot_encode(["dog", "cat", "bird", "dog"], ["cat", "dog", "bird"]) == [
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0]
], "Test 1 Failed"

# Test 2: Danh sách nhãn rỗng, kết quả cũng phải là danh sách rỗng
assert one_hot_encode([], ["cat", "dog", "bird"]) == [], "Test 2 Failed"

# Test 3: Trường hợp chỉ có hai lớp, lặp lại nhãn nhiều lần
assert one_hot_encode(["A", "A", "B"], ["A", "B"]) == [
    [1, 0],
    [1, 0],
    [0, 1]
], "Test 3 Failed"

# Test 4: Chỉ có một nhãn duy nhất, kiểm tra đầu ra đơn
assert one_hot_encode(["cat"], ["cat", "dog", "bird"]) == [
    [1, 0, 0]
], "Test 4 Failed"

# Test 5: Lớp có nhãn dài, kiểm tra ánh xạ không bị ảnh hưởng bởi độ dài chuỗi
assert one_hot_encode(["sad", "happy"], ["happy", "sad", "neutral"]) == [
    [0, 1, 0],
    [1, 0, 0]
], "Test 5 Failed"