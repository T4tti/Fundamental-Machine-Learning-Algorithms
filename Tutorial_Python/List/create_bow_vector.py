# Bài 9: Biểu Diễn Văn Bản Bằng Túi Từ (Bag-of-Words)
'''
Cho một kho văn bản (corpus) được biểu diễn dưới dạng danh sách các tài liệu (mỗi tài
liệu là một danh sách các từ đã được token hóa) và một từ điển (vocabulary) đã được xây
dựng sẵn. Hãy viết một hàm để:
    • Chuyển đổi mỗi tài liệu trong kho văn bản thành một vector Bag-of-Words.
    • Vector BoW có độ dài bằng kích thước của từ điển.
    • Mỗi phần tử thứ i trong vector biểu diễn số lần từ thứ i trong từ điển xuất hiện
trong tài liệu đó. Các từ không có trong từ điển sẽ bị bỏ qua.
Input:
    • 'corpus': list các list, mỗi list con là một tài liệu chứa các từ.
    • 'vocabulary': list các từ duy nhất, đã được sắp xếp, dùng làm từ điển.
Output: Một list các vector BoW.
Ví dụ:
Input:
corpus = [
["ai", "is", "great"],
["ai", "is", "fun", "and", "ai", "is", "cool"]
]
vocabulary = ["ai", "cool", "fun", "great", "is"]
Output:
[[1, 0, 0, 1, 1], # "ai":1, "is":1, "great":1
[2, 1, 1, 0, 2]] # "ai":2, "is":2, "fun":1, "cool":1
'''

def create_bow_vectors(corpus, vocabulary):
    # Bước 1: Tạo một map (dictionary) từ từ sang chỉ số để tra cứu nhanh
    vocab_map = {word: i for i, word in enumerate(vocabulary)}

    # Bước 2: Khởi tạo danh sách rỗng để chứa các vector BoW
    bow_vectors = []
    # Bước 3: Duyệt qua từng tài liệu (document) trong kho văn bản (corpus)
    for doc in corpus:
        # Khởi tạo một vector 0 có độ dài bằng kích thước từ điển cho tài liệu hiện tại
        vector = [0] * len(vocabulary)
        # Duyệt qua từng từ (token) trong tài liệu
        for token in doc:
            # Nếu từ đó có trong vocab_map:
            if token in vocab_map:
                idx = vocab_map[token]
                vector[idx] += 1
        bow_vectors.append(vector[idx])
    return bow_vectors

