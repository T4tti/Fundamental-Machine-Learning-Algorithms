# Bài 3: Làm Phẳng Token Để Tạo Vocabulary NLP

'''
    Cho một kho văn bản được biểu diễn dưới dạng danh sách các câu, trong đó mỗi câu là
một danh sách các từ (token), hãy viết một hàm thực hiện việc:

    Làm phẳng toàn bộ danh sách lồng nhau thành một danh sách chứa tất cả các từ có
trong kho văn bản, theo đúng thứ tự ban đầu.

    Input: Một list các list, mỗi list con là một danh sách các từ.
    Output: Một list chứa tất cả các từ, theo thứ tự xuất hiện ban đầu.

Ví dụ:
Input: [["I", "love", "AI"], ["NLP", "is", "fun"]]
Output: ["I", "love", "AI", "NLP", "is", "fun"]
'''

def flatten_tokens(corpus):
    flatten = []
    for token in corpus:
        for word in token:
            flatten.append(word)
    return flatten

 # Test 1: Hai câu, mỗi câu có nhiều từ, làm phẳng toàn bộ
assert flatten_tokens([["hello", "world"], ["this", "is", "a", "test"]]) == \
["hello", "world", "this", "is", "a", "test"], "Test 1 Failed"

# Test 2: Một câu ngắn và một câu 1 từ, kiểm tra xử lý danh sách không đều
assert flatten_tokens([["a", "b"], ["c"]]) == ["a", "b", "c"], "Test 2 Failed"

 # Test 3: Kho văn bản rỗng, đầu ra là list rỗng
assert flatten_tokens([]) == [], "Test 3 Failed"

 # Test 4: Chỉ có một câu, kiểm tra hoạt động đơn lẻ
assert flatten_tokens([["single", "sentence"]]) == ["single", "sentence"], \
"Test 4 Failed"

 # Test 5: Nhiều câu có độ dài khác nhau, kiểm tra tính ổn định của kết quả
assert flatten_tokens([["deep", "learning"], ["rocks"], ["NLP", "is", "fun"]]) == \
["deep", "learning", "rocks", "NLP", "is", "fun"], "Test 5 Failed"

rs = flatten_tokens([["hello", "world"], ["this", "is", "a", "test"]])
print(rs)
