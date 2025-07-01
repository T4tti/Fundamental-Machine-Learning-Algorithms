# Bài 2: Phân tách dữ liệu: Train - Validation - Test
def split_dataset(data, train_ratio, val_ratio):
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * val_ratio) + train_end

    train_set = data[:train_end]
    val_set = data[train_end:val_end]
    test_set = data[val_end:]

    return train_set, val_set, test_set

dataset = ['a', 'b', 'c']
train, val, test = split_dataset(dataset, 0.5, 0.5)

print("Số phần tử tập train:", len(train))
print("Số phần tử tập validation:", len(val))
print("Số phần tử tập test:", len(test))
