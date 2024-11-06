import torch # type: ignore
from transformers import AutoTokenizer, AutoModel # type: ignore

# Load the transformer model
model_name = 'vinai/phobert-base'
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def get_embedding(text):
    # Kiểm tra văn bản có rỗng hay không
    if not text.strip():
        return torch.zeros((1, model.config.hidden_size))  # Trả về vector embedding 0
    try:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            # Lấy trung bình các trạng thái cuối
            embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings
    except Exception as e:
        print(f"Error processing text: {text}")
        print(str(e))
        return torch.zeros((1, model.config.hidden_size))  # Trả về vector embedding 0 nếu có lỗi

class ReRanker:
    # có thể thay thế model hoặc tokenizer nếu muốn 
    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer

    def rank(self, query, docs):
        # Tính toán embedding cho query
        query_embedding = get_embedding(query)
        # Tính toán embedding cho tất cả các docs
        doc_embeddings = torch.vstack([get_embedding(doc) for doc in docs])
        
        # Tính độ tương đồng bằng tích vô hướng
        similarities = torch.mm(query_embedding, doc_embeddings.T).flatten()
        
        # Sắp xếp các chỉ số theo độ tương đồng
        ranked_indices = similarities.argsort(descending=True)
        ranked_docs = [docs[i] for i in ranked_indices]
        return ranked_docs
