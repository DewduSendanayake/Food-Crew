# act as the tool for retriever agent

import json
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ✅ Path to FAISS index
INDEX_PATH = "vector_store/food_index"

# ✅ Load embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Load FAISS index
vector_store = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

def search_food_items(query: str, top_k: int = 3):
    """Search FAISS index for relevant food items and return JSON."""
    docs = vector_store.similarity_search(query, k=top_k)
    results = []

    for doc in docs:
        metadata = doc.metadata
        results.append({
            "productId": metadata.get("item_id", ""),
            "name": metadata.get("item_name", ""),
            "description": metadata.get("description", ""),
            "price": metadata.get("price", ""),
            "tags": metadata.get("tags", []),
            "score": doc.metadata.get("score", 0)  # FAISS doesn't return score by default, you can compute later if needed
        })

    return json.dumps({"items": results}, indent=2)
