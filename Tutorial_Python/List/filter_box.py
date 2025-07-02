# Bài 5: Lọc Các Bounding Box Dựa Trên Confidence Score

'''
Cho một danh sách các dự đoán từ mô hình nhận dạng vật thể, trong đó mỗi dự đoán là
một danh sách gồm 6 phần tử theo định dạng:

[class_id, conf idence, x, y, w, h]

Hãy viết một hàm để:
    • Lọc ra và giữ lại các dự đoán có độ tự tin (confidence) lớn hơn hoặc bằng một
ngưỡng cho trước (threshold).

Input:
    • predictions: một list các list, mỗi list con đại diện cho một dự đoán với 6 phần tử
như trên.
    • threshold: một số thực trong đoạn [0, 1], là ngưỡng lọc độ tự tin.

Output: Một list mới chứa các dự đoán có confidence ≥ threshold.
Ví dụ:
Input: [[0, 0.9, 10, 10, 100, 100], [1, 0.3, 15, 15, 80, 80]]
Threshold: 0.5
Output: [[0, 0.9, 10, 10, 100, 100]]
'''

def filter_low_confidence_boxes(predictions, threshold):
    filtered_boxes = []
    for box in predictions:
        confidence = box[1]
        if confidence >= threshold:
            filtered_boxes.append(box)
    return filtered_boxes

# Test 1: Có 2 boxes đủ điểm tự tin (>= 0.8), 1 box bị loại
predictions1 = [
    [0, 0.95, 10, 10, 50, 50],
    [1, 0.4, 20, 20, 30, 30],
    [0, 0.88, 15, 15, 40, 40]
]
assert filter_low_confidence_boxes(predictions1, 0.8) == [
    [0, 0.95, 10, 10, 50, 50],
    [0, 0.88, 15, 15, 40, 40]
], "Test 1 Failed"

# Test 2: Threshold cao hơn mọi boxes -> kết quả rỗng
assert filter_low_confidence_boxes(predictions1, 0.99) == [], "Test 2 Failed"

# Test 3: Dữ liệu đầu vào rỗng -> đầu ra cũng rỗng
assert filter_low_confidence_boxes([], 0.5) == [], "Test 3 Failed"

# Test 4: Một box có confidence đúng bằng threshold -> vẫn được giữ lại
predictions2 = [[0, 0.5, 5, 5, 10, 10]]
assert filter_low_confidence_boxes(predictions2, 0.5) == [[0, 0.5, 5, 5, 10, 10]], "Test 4 Failed"

# Test 5: Tất cả các boxes đều đủ điểm -> không loại boxes nào
predictions3 = [[0, 0.85, 1, 2, 3, 4], [1, 0.95, 4, 5, 6, 7]]
assert filter_low_confidence_boxes(predictions3, 0.5) == predictions3, "Test 5 Failed"

print("Tất cả các test case đã PASSED.")
