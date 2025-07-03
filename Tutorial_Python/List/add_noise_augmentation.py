# Bài 8: Tăng Cường Dữ Liệu Chuỗi Thời Gian

'''
Cho một chuỗi thời gian (được biểu diễn dưới dạng một list các số) và một độ lệch chuẩn
của nhiễu, hãy viết một hàm thực hiện:
    • Tạo ra một chuỗi thời gian mới có cùng độ dài.
    • Mỗi điểm dữ liệu trong chuỗi mới bằng điểm dữ liệu tương ứng trong chuỗi gốc cộng
với một giá trị nhiễu ngẫu nhiên.
    • Giá trị nhiễu này được lấy mẫu từ một phân phối chuẩn (Gaussian distribution) với
giá trị trung bình là 0 và độ lệch chuẩn ('noise_level') cho trước.
Input:
    • 'time_series': một list các số.
    • 'noise_level': một số thực không âm, đại diện cho độ lệch chuẩn của nhiễu.
Output: Một list mới là chuỗi thời gian đã được tăng cường bằng nhiễu.
*Gợi ý: Sử dụng thư viện 'random' của Python, cụ thể là hàm 'random.gauss(mu, sigma)'.*
'''
import random
def add_noise_augmentation(time_series, noise_level):
    # Bước 1: Khởi tạo danh sách kết quả rỗng
    noisy_series = []

    # Bước 2: Duyệt qua từng điểm dữ liệu trong chuỗi thời gian gốc
    for value in time_series:
        # Tạo một giá trị nhiễu ngẫu nhiên từ phân phối Gaussian (mu=0, sigma=noise_level)
        noise = random.gauss(0, noise_level)
        # Tính giá trị mới bằng cách cộng điểm dữ liệu gốc với nhiễu
        augmented_value = value + noise
        # Thêm giá trị mới vào danh sách kết quả
        noisy_series.append(augmented_value)
    
    return noisy_series

# ==============================
# TEST CASES
# ==============================

random.seed(0)

# Test 1: Thêm nhiễu vào một chuỗi thời gian
ts1 = [10, 11, 12, 11, 10]
augmented_ts1 = add_noise_augmentation(ts1, 0.1)
assert len(augmented_ts1) == len(ts1), "Test 1 Failed: Length mismatch"
assert augmented_ts1 != ts1, \
    "Test 1 Failed: Series should be different after adding noise"

# Test 2: noise_level = 0, chuỗi không thay đổi
ts2 = [100, 200, 150]
augmented_ts2 = add_noise_augmentation(ts2, 0.0)
assert augmented_ts2 == ts2, \
    "Test 2 Failed: Series should be identical with zero noise"

# Test 3: Chuỗi rỗng
assert add_noise_augmentation([], 0.5) == [], \
    "Test 3 Failed: Empty list should return empty list"

# Test 4: Chuỗi một phần tử phải khác sau khi thêm nhiễu
ts4 = [5]
augmented_ts4 = add_noise_augmentation(ts4, 1.0)
assert augmented_ts4 != ts4, "Test 4 Failed: Single element should change"

print("All tests passed successfully.")

