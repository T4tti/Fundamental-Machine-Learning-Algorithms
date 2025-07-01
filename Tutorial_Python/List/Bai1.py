# Tutorial: Ứng dụng của List trong các bài toán thực tế
# Bài 1: Chuẩn hóa độ dài chuỗi đầu vào bằng kỹ thuật Padding

def pad_sequences(sequences):
    """
    Chuẩn hóa độ dài của các chuỗi trong danh sách bằng cách thêm ký tự '0' vào đầu chuỗi.
    
    Args:
        sequences (list of str): Danh sách các chuỗi đầu vào.
        
    Returns:
        list of str: Danh sách các chuỗi đã được chuẩn hóa độ dài.
    """
    max_length = max(len(seq) for seq in sequences)
    pad_sequences = []

    for seq in sequences:
        num_padding = max_length - len(seq)
        padded_seq = seq + [0] * num_padding
        pad_sequences.append(padded_seq)
    return pad_sequences

# Ví dụ sử dụng hàm pad_sequences
ds = [[1, 2, 3], [4, 5, 0], [6, 0, 0]]
kq = pad_sequences(ds)
print(kq)

