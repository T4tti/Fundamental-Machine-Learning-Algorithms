# Bài 2: Phân tách dữ liệu: Train - Validation - Test
def split_dataset(data, train_ratio, val_ratio):
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * val_ratio) + train_end

    train_set = data[:train_end]
    val_set = data[train_end:val_end]
    test_set = data[val_end:]

    return train_set, val_set, test_set


# Test 1: Chia tỷ lệ chuẩn
dataset = list(range(100))
train, val, test = split_dataset(dataset, 0.7, 0.15)
assert (len(train), len(val), len(test)) == (70, 15, 15), "Test 1 Failed"

# Test 2: Dữ liệu nhỏ, chia đều, kiểm tra làm tròn
dataset = [’a’, ’b’, ’c’]
train, val, test = split_dataset(dataset, 0.5, 0.5)
assert (len(train), len(val), len(test)) == (1, 1, 1), "Test 2 Failed"

# Test 3: Không có tập validation
dataset = list(range(5))
train, val, test = split_dataset(dataset, 0.8, 0.0)
assert (train, val, test) == (list(range(4)), [], [4]), "Test 3 Failed"

# Test 4: Dữ liệu rỗng
dataset = []
train, val, test = split_dataset(dataset, 0.6, 0.2)
assert (train, val, test) == ([], [], []), "Test 4 Failed"

# Test 5: Tổng tỉ lệ nhỏ hơn 1, phần còn lại là test
dataset = list(range(10))
train, val, test = split_dataset(dataset, 0.2, 0.3) # 2, 3, 5
assert (train, val, test) == \
(list(range(2)), list(range(2, 5)), list(range(5, 10))), "Test 5 Failed"
