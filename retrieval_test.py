from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

print("Loading embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings,
    collection_name="infosys_milestone1"
)

print("Vector DB Loaded Successfully!")

query = input("Enter your question: ")

results = vectorstore.similarity_search(query, k=3)

print("\nTop Retrieved Results:\n")

for i, doc in enumerate(results):
    print(f"\nResult {i+1}:")
    print(doc.page_content)
    print("-" * 50)