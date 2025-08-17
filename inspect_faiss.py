#just checking index.faiss and index.pkl

import os
import faiss
import pickle
import json

# Path to your FAISS index folder
index_folder = "vector_store/food_index"
faiss_file = os.path.join(index_folder, "index.faiss")
pkl_file = os.path.join(index_folder, "index.pkl")
json_file = os.path.join(index_folder, "docstore.json")

# 1. Load FAISS index
print("Loading FAISS index...")
if os.path.exists(faiss_file):
    index = faiss.read_index(faiss_file)
    print(f"✅ FAISS index loaded.")
    print(f"Number of vectors: {index.ntotal}")
    print(f"Index type: {type(index)}")
else:
    print("❌ index.faiss not found.")
    index = None

# 2. Load associated metadata (LangChain style)
if os.path.exists(pkl_file):
    print("\nLoading metadata from index.pkl...")
    with open(pkl_file, "rb") as f:
        pkl_data = pickle.load(f)
    print(f"✅ Loaded index.pkl. Type: {type(pkl_data)}")
    print("Keys in pickle file:", list(pkl_data.keys()) if isinstance(pkl_data, dict) else "Not a dict")
else:
    print("\nindex.pkl not found.")

if os.path.exists(json_file):
    print("\nLoading documents from docstore.json...")
    with open(json_file, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    print(f"✅ Loaded docstore.json. Found {len(json_data)} entries.")
    # Print a few entries
    for i, (k, v) in enumerate(json_data.items()):
        print(f"\nDocument ID: {k}")
        print(f"Content: {v.get('page_content', '')[:200]}...")
        print(f"Metadata: {v.get('metadata', {})}")
        if i >= 2:  # limit to first 3 docs
            break
else:
    print("\ndocstore.json not found.")

# 3. If FAISS index is loaded, print vector shape
if index is not None:
    try:
        # Reconstruct all vectors (be careful if too large)
        print("\nExtracting vectors...")
        vectors = []
        for i in range(index.ntotal):
            vectors.append(index.reconstruct(i))
        print(f"✅ Extracted {len(vectors)} vectors. Example vector shape: {len(vectors[0])}")
    except Exception as e:
        print("Error extracting vectors:", e)
