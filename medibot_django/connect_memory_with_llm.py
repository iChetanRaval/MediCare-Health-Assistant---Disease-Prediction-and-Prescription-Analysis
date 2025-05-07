# import os

# from langchain_huggingface import HuggingFaceEndpoint
# from langchain_core.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS

# ## Uncomment the following files if you're not using pipenv as your virtual environment manager
# from dotenv import load_dotenv, find_dotenv
# load_dotenv(find_dotenv())

# # Step 1: Setup LLM (Mistral with HuggingFace)
# HF_TOKEN=os.environ.get("HF_TOKEN")
# HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"

# # def load_llm(huggingface_repo_id):
# #     llm=HuggingFaceEndpoint(
# #         repo_id=huggingface_repo_id,
# #         temperature=0.5,
# #         model_kwargs={"token":HF_TOKEN,
# #                       "max_length":"512"}
# #     )
# #     return llm
# def load_llm(huggingface_repo_id):
#     llm = HuggingFaceEndpoint(
#         repo_id=huggingface_repo_id,
#         temperature=0.5,
#         model_kwargs={"use_auth_token": HF_TOKEN,  # Using the correct token key
#                       "max_length": 512}  # Adjusted max_length
#     )
#     return llm


# # Step 2: Connect LLM with FAISS and Create chain

# CUSTOM_PROMPT_TEMPLATE = """
# Use the pieces of information provided in the context to answer user's question.
# If you dont know the answer, just say that you dont know, dont try to make up an answer. 
# Dont provide anything out of the given context

# Context: {context}
# Question: {question}

# Start the answer directly. No small talk please.
# """

# def set_custom_prompt(custom_prompt_template):
#     prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
#     return prompt

# # Load Database
# DB_FAISS_PATH="vectorstore/db_faiss"
# embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# # Create QA chain
# qa_chain=RetrievalQA.from_chain_type(
#     llm=load_llm(HUGGINGFACE_REPO_ID),
#     chain_type="stuff",
#     retriever=db.as_retriever(search_kwargs={'k':3}),
#     return_source_documents=True,
#     chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
# )

# # Now invoke with a single query
# user_query=input("Write Query Here: ")
# response=qa_chain.invoke({'query': user_query})
# print("RESULT: ", response["result"])
# print("SOURCE DOCUMENTS: ", response["source_documents"])

# connect_memory_with_llm.py

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Setup
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# Step 1: Load LLM
def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={
            "use_auth_token": HF_TOKEN,  # Correct token key
            "max_length": 512
        },
        task="text-generation"  # Set task to text-generation for this model
    )
    return llm


# Step 2: Set up the custom prompt
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know. Don't try to make up an answer. 
Don't provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk, please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Step 3: Load FAISS database
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Step 4: Create the QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 5}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Step 5: Query the chain
user_query = input("Write Query Here: ")
response = qa_chain.invoke({'query': user_query})

# Output result
print("RESULT: ", response["result"])
print("SOURCE DOCUMENTS: ", response["source_documents"])
