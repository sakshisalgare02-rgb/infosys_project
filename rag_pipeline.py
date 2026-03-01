import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# ---------------------------
# 1. LOAD EMBEDDINGS
# ---------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------------------
# 2. LOAD VECTOR DATABASE
# ---------------------------
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings,
    collection_name="infosys_milestone1"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ---------------------------
# 3. CONNECT LLM (DIRECT API KEY)
# ---------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key="GROQ_API_KEY"
)

# ---------------------------
# 4. PROMPT TEMPLATE
# ---------------------------
prompt = ChatPromptTemplate.from_template("""
You are an AI assistant.
Answer ONLY from the provided context.
If answer not found, say "I don't know".

Context:
{context}

Question:
{question}
""")

print("✅ RAG Pipeline Ready")

# ---------------------------
# 5. USER QUERY
# ---------------------------
query = input("Enter your question: ")

docs = retriever.invoke(query)

context = "\n\n".join([doc.page_content for doc in docs])

final_prompt = prompt.invoke({
    "context": context,
    "question": query
})

response = llm.invoke(final_prompt)

print("\n💡 Answer:\n")
print(response.content)

print("\n📚 Sources:")
for i, doc in enumerate(docs):
    print(f"Source {i+1}: {doc.metadata}")