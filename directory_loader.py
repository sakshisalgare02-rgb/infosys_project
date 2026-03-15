import os
import shutil
from dotenv import load_dotenv

# ----------------------------
# LOAD ENV VARIABLES
# ----------------------------
load_dotenv()

# ----------------------------
# IMPORTS
# ----------------------------
from langchain_groq import ChatGroq
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

print("Starting program...")

# ----------------------------
# GROQ MODEL INIT
# ----------------------------
model = ChatGroq(model="llama-3.1-8b-instant")
print("Groq model initialized successfully.")

# ----------------------------
# DELETE OLD DATABASE
# ----------------------------
if os.path.exists("chroma_db"):
    shutil.rmtree("chroma_db")
    print("Old database deleted.")

# ----------------------------
# LOAD PDF FILES
# ----------------------------
pdf_loader = DirectoryLoader(
    "Books",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

pdf_docs = pdf_loader.load()

# ----------------------------
# LOAD TEXT FILES
# ----------------------------
text_loader = DirectoryLoader(
    "Books",
    glob="*.txt",
    loader_cls=TextLoader
)

text_docs = text_loader.load()

# Combine documents
docs = pdf_docs + text_docs
print(f"Total documents loaded: {len(docs)}")

# Stop if no files found
if len(docs) == 0:
    print("❌ No PDF or TXT files found inside Books folder.")
    exit()

# ----------------------------
# TEXT SPLITTING
# ----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=200
)

split_docs = text_splitter.split_documents(docs)
print(f"Total chunks created: {len(split_docs)}")

# ----------------------------
# CLEAN METADATA
# ----------------------------
for doc in split_docs:
    doc.metadata = {}

# ----------------------------
# EMBEDDINGS
# ----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ----------------------------
# CREATE VECTOR DATABASE
# ----------------------------
vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory="chroma_db",
    collection_name="infosys_milestone1"
)

vectorstore.persist()

print("✅ Database successfully created and saved.")