from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ✅ Load embeddings (must be same model used for indexing)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Load FAISS index
vectorstore = FAISS.load_local("vector_store/food_index", embeddings, allow_dangerous_deserialization=True)

# ✅ Create a retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # top 3 results

# ✅ Test query
query = "I want something spicy with chicken"
results = retriever.get_relevant_documents(query)

print("\n=== Top Matches ===\n")
for idx, doc in enumerate(results, start=1):
    print(f"{idx}. {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print("-" * 50)
