# #Load raw PDF
# #Create Chunks
# #Create vectors Embedding
# #Store embedding in FAISS
# from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS

# #Step 1: Load raw PDF
# DATA_PATH="data/"
# def load_pdf_files(data):
#   loader = DirectoryLoader(data,glob ="*.pdf",loader_cls=PyPDFLoader)

#   documents = loader.load()
#   return documents

# documents = load_pdf_files(data=DATA_PATH)
# # print("Length of documents pages",len(documents))

# #Step 2: Create Chunks
# def create_chunks(extracted_data):
#   text_splitter = RecursiveCharacterTextSplitter(chunk_size=600,chunk_overlap=100)
#   text_chunks = text_splitter.split_documents(extracted_data)
#   return text_chunks

# text_chunks=create_chunks(extracted_data=documents)
# # print("Length of Text chunks",len(text_chunks))

# #Step 3: Create vectors Embedding
# def get_embedding_model():
#   embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#   return embedding_model

# embedding_model = get_embedding_model()

# #Step 4: Store embedding in FAISS
# DB_FAISS_PATH = "vectorstore/db_faiss"
# db= FAISS.from_documents(text_chunks,embedding_model)
# db.save_local(DB_FAISS_PATH)

# create_memory_for_LLM.py 

# from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# import os

# # Step 1: Load raw PDF files efficiently
# DATA_PATH = "data/"

# def load_pdf_files(data_path):
#     loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
#     documents = loader.load()
#     return documents

# documents = load_pdf_files(DATA_PATH)
# print("Loaded", len(documents), "pages.")

# # Step 2: Create Chunks (Optimized chunk size)
# def create_chunks(extracted_data, chunk_size=1000, chunk_overlap=200):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#     text_chunks = text_splitter.split_documents(extracted_data)
#     return text_chunks

# text_chunks = create_chunks(documents)
# print("Generated", len(text_chunks), "text chunks.")

# # Step 3: Get Embedding Model (Use a Faster Model if Needed)
# def get_embedding_model():
#     return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# embedding_model = get_embedding_model()

# # Step 4: Store embeddings in FAISS (Optimized)
# DB_FAISS_PATH = "vectorstore/db_faiss"

# if not os.path.exists(DB_FAISS_PATH):  
#     os.makedirs(DB_FAISS_PATH)

# db = FAISS.from_documents(text_chunks, embedding_model)
# db.save_local(DB_FAISS_PATH)

# print("FAISS vector database saved successfully.")

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# Step 1: Load raw PDF files efficiently
DATA_PATH = "data/"

def load_pdf_files(data_path):
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

documents = load_pdf_files(DATA_PATH)
print("Loaded", len(documents), "pages.")

# Step 2: Create Chunks (Optimized chunk size)
def create_chunks(extracted_data, chunk_size=2000, chunk_overlap=500):  # Increase chunk_size
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = create_chunks(documents)
print("Generated", len(text_chunks), "text chunks.")

# Step 3: Get Embedding Model (Use a Faster Model if Needed)
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embedding_model = get_embedding_model()

# Step 4: Store embeddings in FAISS (Process in batches)
DB_FAISS_PATH = "vectorstore/db_faiss"

if not os.path.exists(DB_FAISS_PATH):  
    os.makedirs(DB_FAISS_PATH)

# Process chunks in batches
batch_size = 10000  # Adjust based on your system's memory
for i in range(0, len(text_chunks), batch_size):
    batch_chunks = text_chunks[i:i + batch_size]
    if i == 0:
        db = FAISS.from_documents(batch_chunks, embedding_model)
    else:
        db.add_documents(batch_chunks)
    print(f"Processed {i + len(batch_chunks)}/{len(text_chunks)} chunks.")

db.save_local(DB_FAISS_PATH)
print("FAISS vector database saved successfully.")