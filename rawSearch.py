import numpy as np # type: ignore
import json
import os  
from collections import Counter

# Lớp TFIDF để tính toán trọng số TF-IDF cho các tài liệu
class TFIDF:
    def __init__(self, documents):
        self.documents = documents
        self.word_freqs = [Counter(doc.split()) for doc in documents]  # Tính tần suất từ cho từng tài liệu
        self.idf = self.calculate_idf()  # Tính IDF cho tất cả các từ

    def calculate_idf(self):
        N = len(self.documents)
        idf = {}
        for doc in self.word_freqs:
            for word in doc.keys():
                if word not in idf:
                    # Đếm số tài liệu chứa từ t
                    df = sum(1 for d in self.word_freqs if word in d)
                    # IDF: IDF(t) = log10(N / df_t) nếu df_t > 0, ngược lại 0
                    idf[word] = np.log10(N / df) if df > 0 else 0
        return idf

    def tf(self, word, doc):
        count_t_d = doc[word]
        # TF(t, d) = 1 + log10(count(t, d)) nếu count(t, d) > 0, ngược lại 0
        return 1 + np.log10(count_t_d) if count_t_d > 0 else 0

    def weight(self, word, doc):
        # w(t, d) = tf(t, d) * idf(t)
        return self.tf(word, doc) * self.idf.get(word, 0)

    def tfidf(self, doc):
        # Tính trọng số TF-IDF cho từng từ trong tài liệu
        return {word: self.weight(word, doc) for word in doc.keys()}

    def search(self, query, k):
        # Chia truy vấn thành các từ
        query_words = query.split()
        # Tính TF-IDF cho truy vấn
        query_tfidf = self.tfidf(Counter(query_words))
        similarities = []
        for doc in self.word_freqs:
            # Tính TF-IDF cho tài liệu
            doc_tfidf = self.tfidf(doc)
            # Tính độ tương đồng giữa truy vấn và tài liệu
            similarity = sum(query_tfidf.get(word, 0) * doc_tfidf.get(word, 0) for word in query_tfidf.keys())
            similarities.append(similarity)

        # Sắp xếp, lấy ra k văn bản có score cao nhất
        top_indices = np.argsort(similarities)[::-1][:k]
        return [self.documents[i] for i in top_indices if similarities[i] > 0]

    def save_tfidf_to_file(self, filename='tf_idf.json'):
        # Lưu dữ liệu TF-IDF vào file
        tfidf_data = {
            'idf': self.idf,
            'word_freqs': [{word: count for word, count in doc.items()} for doc in self.word_freqs]
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(tfidf_data, f, ensure_ascii=False, indent=4)

    def load_tfidf_from_file(self, filename='tf_idf.json'):
        # Đọc dữ liệu TF-IDF từ file
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.idf = data['idf']
                self.word_freqs = [Counter(doc) for doc in data['word_freqs']]

                
