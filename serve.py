import json
import numpy as np # type: ignore
from flask import Flask, request, jsonify # type: ignore
import pandas as pd # type: ignore
from rawSearch import TFIDF
from reranker import ReRanker
from collections import Counter

app = Flask(__name__)

# Load documents from JSON
file_path = 'docs.json'  # Đường dẫn đến file chứa dữ liệu
data = pd.read_json(file_path, encoding='utf-8')
documents = data['docs'].tolist()  # Lấy danh sách tài liệu
tf_idf = TFIDF(documents)
tf_idf.save_tfidf_to_file()  # Lưu TF-IDF khi khởi động

@app.route('/api/search', methods=['POST'])
def handle_query():
    data = request.get_json()

    # Kiểm tra định dạng yêu cầu
    if not data or not isinstance(data, dict):
        return jsonify({'response': 'Invalid request format.'}), 400

    # Kiểm tra sự tồn tại của khóa 'query'
    if 'query' not in data:
        return jsonify({'response': 'No key in dict.'}), 400

    query = data['query']  # Lấy giá trị từ khóa 'query'
    
    # Kiểm tra nếu query rỗng hoặc không hợp lệ
    if not query or not isinstance(query, str) or not query.strip():
        return jsonify({'response': 'None query.'}), 400

    clean_query = query.lower().strip()  # Làm sạch truy vấn
    print(f"Processing query: {clean_query}")

    # Đọc TF-IDF từ file
    tf_idf.load_tfidf_from_file()

    # Tính điểm cho từng tài liệu dựa trên TF-IDF đã lưu
    query_tfidf = tf_idf.tfidf(Counter(clean_query.split()))
    similarities = []
    for doc in tf_idf.word_freqs:
        doc_tfidf = tf_idf.tfidf(doc)
        # Tính độ tương đồng giữa truy vấn và tài liệu
        similarity = sum(query_tfidf.get(word, 0) * doc_tfidf.get(word, 0) for word in query_tfidf.keys())
        similarities.append(similarity)

    # Lưu điểm vào score.json
    scores = {tf_idf.documents[i]: similarities[i] for i in range(len(similarities))}
    with open('score.json', 'w', encoding='utf-8') as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)

    # Lấy 10 tài liệu có điểm cao nhất
    top_indices = np.argsort(similarities)[::-1][:10]
    filtered_results = [tf_idf.documents[i] for i in top_indices if similarities[i] > 0]

    if not filtered_results:
        return jsonify({'response': 'Cannot find results for the provided query.'}), 404

    # Re-rank sử dụng AI
    re_ranker = ReRanker()
    ranked_docs = re_ranker.rank(clean_query, filtered_results)

    # Lưu kết quả vào results.json
    results = [{"rank": rank, "document": doc} for rank, doc in enumerate(ranked_docs, start=1)]
    with open('results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    return jsonify({'response': ranked_docs})

if __name__ == '__main__':
    app.run(debug=True)  # Chạy ứng dụng Flask
