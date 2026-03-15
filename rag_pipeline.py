import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# ---------------------------
# LOAD EMBEDDINGS
# ---------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------------------
# LOAD VECTOR DATABASE
# ---------------------------
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings,
    collection_name="infosys_milestone1"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ---------------------------
# CONNECT LLM (API KEY FROM .env)
# ---------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

# ---------------------------
# PROMPT TEMPLATE
# ---------------------------
prompt = ChatPromptTemplate.from_template("""
You are an AI assistant.
Answer ONLY from the provided context.
If the answer is not found, say "I don't know".

Context:
{context}

Question:
{question}
""")

# ---------------------------
# MAIN FUNCTION
# ---------------------------
def ask_question(query):

    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    response = llm.invoke(final_prompt)

    return response.content