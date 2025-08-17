import os
import json
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# ✅ Load JSON from file
with open("data/menu_items.json", "r", encoding="utf-8") as f:
    test_items = json.load(f)

# ✅ Convert each item into a Document
docs = []
for item in test_items["items"]:
    name = item.get("name") or item.get("item_name", "")
    alternate_name = item.get("alternateName", "")
    description = item.get("description", "")
    tags = ", ".join(item.get("tags", []))
    price = item.get("price", "")
    text = f"""
    Name: {name or alternate_name}
    Alternate Name: {alternate_name}
    Description: {description}
    Tags: {tags}
    Price: {price}
    """
    # Store full item as metadata
    docs.append(Document(page_content=text.strip(), metadata=item))

# ✅ Load local embedding model (free)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Create FAISS index
vectorstore = FAISS.from_documents(docs, embeddings)

# ✅ Save index to disk
os.makedirs("vector_store", exist_ok=True)
vectorstore.save_local("vector_store/food_index")

print("✅ FAISS index built and saved at vector_store/food_index")
