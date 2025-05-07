#Main   --------working--------
# from django.shortcuts import render
# from django.http import JsonResponse
# from langchain_huggingface import HuggingFaceEndpoint
# from langchain_core.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# import os
# from dotenv import load_dotenv, find_dotenv

# load_dotenv(find_dotenv())

# # Setup
# HF_TOKEN = os.environ.get("HF_TOKEN")
# HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# # Step 1: Load LLM
# def load_llm(huggingface_repo_id):
#     llm = HuggingFaceEndpoint(
#         repo_id=huggingface_repo_id,
#         temperature=0.5,
#         model_kwargs={
#             "use_auth_token": HF_TOKEN,  # Correct token key
#             "max_length": 512
#         },
#         task="text-generation"  # Set task to text-generation for this model
#     )
#     return llm

# # Step 2: Set up the custom prompt
# CUSTOM_PROMPT_TEMPLATE = """
# Use the pieces of information provided in the context to answer the user's question.
# If you don't know the answer, just say that you don't know. Don't try to make up an answer. 
# Don't provide anything out of the given context.

# Context: {context}
# Question: {question}

# Start the answer directly. No small talk, please.
# """

# def set_custom_prompt(custom_prompt_template):
#     prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
#     return prompt

# # Step 3: Load FAISS database
# DB_FAISS_PATH = "vectorstore/db_faiss"
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# # Step 4: Create the QA chain
# qa_chain = RetrievalQA.from_chain_type(
#     llm=load_llm(HUGGINGFACE_REPO_ID),
#     chain_type="stuff",
#     retriever=db.as_retriever(search_kwargs={'k': 5}),
#     return_source_documents=True,
#     chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
# )

# def chatbot_response(request):
#     response = None
#     if request.method == 'POST':
#         prompt = request.POST.get('prompt')
#         response = qa_chain.invoke({'query': prompt})
#         response = response["result"]

#     return render(request, 'chatbot/index.html', {'response': response})



# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# import json
# from langchain_huggingface import HuggingFaceEndpoint
# from langchain_core.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# import os
# from dotenv import load_dotenv, find_dotenv

# load_dotenv(find_dotenv())

# # Setup
# HF_TOKEN = os.environ.get("HF_TOKEN")
# HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# # Step 1: Load LLM
# def load_llm(huggingface_repo_id):
#     llm = HuggingFaceEndpoint(
#         repo_id=huggingface_repo_id,
#         temperature=0.5,
#         model_kwargs={
#             "use_auth_token": HF_TOKEN,  # Correct token key
#             "max_length": 512
#         },
#         task="text-generation"  # Set task to text-generation for this model
#     )
#     return llm

# # Step 2: Set up the custom prompt
# CUSTOM_PROMPT_TEMPLATE = """
# Use the pieces of information provided in the context to answer the user's question.
# If you don't know the answer, just say that you don't know. Don't try to make up an answer. 
# Don't provide anything out of the given context.

# Context: {context}
# Question: {question}

# Start the answer directly. No small talk, please.
# """

# def set_custom_prompt(custom_prompt_template):
#     prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
#     return prompt

# # Step 3: Load FAISS database
# DB_FAISS_PATH = "vectorstore/db_faiss"
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# # Step 4: Create the QA chain
# qa_chain = RetrievalQA.from_chain_type(
#     llm=load_llm(HUGGINGFACE_REPO_ID),
#     chain_type="stuff",
#     retriever=db.as_retriever(search_kwargs={'k': 5}),
#     return_source_documents=True,
#     chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
# )

# @csrf_exempt  # Allow POST requests without CSRF token (for simplicity, use proper CSRF handling in production)
# def chatbot_response(request):
#     if request.method == 'POST':
#         try:
#             data = json.loads(request.body)
#             prompt = data.get('prompt')
            
#             # Process the query using the QA chain
#             response = qa_chain.invoke({'query': prompt})
#             result = response["result"]
            
#             return JsonResponse({'response': result})
#         except Exception as e:
#             return JsonResponse({'error': str(e)}, status=500)
#     return JsonResponse({'error': 'Invalid request method'}, status=400)



# from django.shortcuts import render
# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# import json
# from langchain_huggingface import HuggingFaceEndpoint
# from langchain_core.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# import os
# from dotenv import load_dotenv, find_dotenv

# load_dotenv(find_dotenv())

# # Setup
# HF_TOKEN = os.environ.get("HF_TOKEN")
# HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# # Step 1: Load LLM
# def load_llm(huggingface_repo_id):
#     llm = HuggingFaceEndpoint(
#         repo_id=huggingface_repo_id,
#         temperature=0.5,
#         model_kwargs={
#             "use_auth_token": HF_TOKEN,  # Correct token key
#             "max_length": 512
#         },
#         task="text-generation"  # Set task to text-generation for this model
#     )
#     return llm

# # Step 2: Set up the custom prompt
# CUSTOM_PROMPT_TEMPLATE = """
# Use the pieces of information provided in the context to answer the user's question.
# If you don't know the answer, just say that you don't know. Don't try to make up an answer. 
# Don't provide anything out of the given context.

# Context: {context}
# Question: {question}

# Start the answer directly. No small talk, please.
# """

# def set_custom_prompt(custom_prompt_template):
#     prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
#     return prompt

# # Step 3: Load FAISS database
# DB_FAISS_PATH = "vectorstore/db_faiss"
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# # Step 4: Create the QA chain
# qa_chain = RetrievalQA.from_chain_type(
#     llm=load_llm(HUGGINGFACE_REPO_ID),
#     chain_type="stuff",
#     retriever=db.as_retriever(search_kwargs={'k': 5}),
#     return_source_documents=True,
#     chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
# )

# # @csrf_exempt  # Allow POST requests without CSRF token (for simplicity, use proper CSRF handling in production)
# # def chatbot_response(request):
# #     if request.method == 'POST':
# #         try:
# #             data = json.loads(request.body)
# #             prompt = data.get('prompt')
            
# #             # Process the query using the QA chain
# #             response = qa_chain.invoke({'query': prompt})
# #             result = response["result"]
            
# #             return JsonResponse({'response': result})
# #         except Exception as e:
# #             return JsonResponse({'error': str(e)}, status=500)
# #     elif request.method == 'GET':
# #         # Render the HTML template for GET requests
# #         return render(request, 'chatbot/index.html')
# #     return JsonResponse({'error': 'Invalid request method'}, status=400)

# @csrf_exempt
# def chatbot_response(request):
#     if request.method == 'POST':
#         try:
#             data = json.loads(request.body)
#             prompt = data.get('prompt')
#             print("User Prompt:", prompt)  # Debug: Print the user's prompt

#             # Process the query using the QA chain
#             response = qa_chain.invoke({'query': prompt})
#             print("QA Chain Response:", response)  # Debug: Print the full response

#             result = response["result"]
#             print("Result:", result)  # Debug: Print the result

#             return JsonResponse({'response': result})
#         except Exception as e:
#             print("Error:", str(e))  # Debug: Print the error
#             return JsonResponse({'error': str(e)}, status=500)
#     elif request.method == 'GET':
#         return render(request, 'chatbot/index.html')
#     return JsonResponse({'error': 'Invalid request method'}, status=400)














# ++++++++1st Level ++++++

# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# import json
# from langchain_huggingface import HuggingFaceEndpoint
# from langchain_core.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# import os
# from django.shortcuts import render
# from dotenv import load_dotenv, find_dotenv

# load_dotenv(find_dotenv())

# # Setup
# HF_TOKEN = os.environ.get("HF_TOKEN")
# HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# # Step 1: Load LLM
# def load_llm(huggingface_repo_id):
#     llm = HuggingFaceEndpoint(
#         repo_id=huggingface_repo_id,
#         temperature=0.5,
#         model_kwargs={
#             "use_auth_token": HF_TOKEN,  # Correct token key
#             "max_length": 512
#         },
#         task="text-generation"  # Set task to text-generation for this model
#     )
#     return llm

# # Step 2: Set up the custom prompt
# CUSTOM_PROMPT_TEMPLATE = """
# Use the pieces of information provided in the context to answer the user's question.
# If you don't know the answer, just say that you don't know. Don't try to make up an answer. 
# Don't provide anything out of the given context.

# Context: {context}
# Question: {question}

# Start the answer directly. No small talk, please.
# """

# def set_custom_prompt(custom_prompt_template):
#     prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
#     return prompt

# # Step 3: Load FAISS database
# DB_FAISS_PATH = "vectorstore/db_faiss"
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# # Step 4: Create the QA chain
# qa_chain = RetrievalQA.from_chain_type(
#     llm=load_llm(HUGGINGFACE_REPO_ID),
#     chain_type="stuff",
#     retriever=db.as_retriever(search_kwargs={'k': 5}),
#     return_source_documents=True,
#     chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
# )

# @csrf_exempt  # Allow POST requests without CSRF token (for simplicity, use proper CSRF handling in production)
# def chatbot_response(request):
#     if request.method == 'POST':
#         try:
#             data = json.loads(request.body)
#             prompt = data.get('prompt')
            
#             # Process the query using the QA chain
#             response = qa_chain.invoke({'query': prompt})
#             result = response["result"]
            
#             return JsonResponse({'response': result})
#         except Exception as e:
#             return JsonResponse({'error': str(e)}, status=500)
#     elif request.method == 'GET':
#         return render(request, 'chatbot/index.html')
#     return JsonResponse({'error': 'Invalid request method'}, status=400)




# ==========working 2nd level ============

# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# import json
# from huggingface_hub import InferenceClient
# from langchain_core.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# import os
# from django.shortcuts import render
# from dotenv import load_dotenv, find_dotenv

# load_dotenv(find_dotenv())

# # Setup
# HF_TOKEN = os.environ.get("HF_TOKEN")
# HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# # Initialize Hugging Face Inference Client
# client = InferenceClient(model=HUGGINGFACE_REPO_ID, token=HF_TOKEN)

# # Custom prompt template
# CUSTOM_PROMPT_TEMPLATE = """
# Use the pieces of information provided in the context to answer the user's question.
# If you don't know the answer, just say that you don't know. Don't try to make up an answer. 
# Don't provide anything out of the given context.

# Context: {context}
# Question: {question}

# Start the answer directly. No small talk, please.
# """

# def set_custom_prompt(custom_prompt_template):
#     return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

# # Load FAISS database
# DB_FAISS_PATH = "vectorstore/db_faiss"
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# # QA Chain using InferenceClient
# def get_qa_response(query):
#     # Retrieve top documents
#     docs = db.similarity_search(query, k=5)
#     context = " ".join([doc.page_content for doc in docs])

#     # Generate response using InferenceClient
#     response = client.text_generation(
#         f"Context: {context}\nQuestion: {query}\nAnswer:",
#         max_new_tokens=200,
#         temperature=0.5
#     )
#     return response

# @csrf_exempt
# def chatbot_response(request):
#     if request.method == 'POST':
#         try:
#             data = json.loads(request.body)
#             prompt = data.get('prompt')

#             # Process the query using the new method
#             result = get_qa_response(prompt)
#             return JsonResponse({'response': result})
#         except Exception as e:
#             return JsonResponse({'error': str(e)}, status=500)
#     elif request.method == 'GET':
#         return render(request, 'chatbot/index.html')
#     return JsonResponse({'error': 'Invalid request method'}, status=400)





# +++++++++ 3rd Level ++++++++


# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# import json
# from langchain_huggingface import HuggingFaceEndpoint
# from langchain_core.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# import os
# from django.shortcuts import render
# from dotenv import load_dotenv, find_dotenv

# load_dotenv(find_dotenv())

# HF_TOKEN = os.environ.get("HF_TOKEN")
# HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# def load_llm(huggingface_repo_id):
#     llm = HuggingFaceEndpoint(
#         repo_id=huggingface_repo_id,
#         temperature=0.5,
#         model_kwargs={"use_auth_token": HF_TOKEN, "max_length": 512},
#         task="text-generation"
#     )
#     return llm

# CUSTOM_PROMPT_TEMPLATE = """
# Use the pieces of information provided in the context to answer the user's question.
# If you don't know the answer, just say that you don't know. Don't try to make up an answer. 
# Don't provide anything out of the given context.

# Context: {context}
# Question: {question}

# Start the answer directly. No small talk, please.
# """

# def set_custom_prompt(custom_prompt_template):
#     return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

# DB_FAISS_PATH = "vectorstore/db_faiss"
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# qa_chain = RetrievalQA.from_chain_type(
#     llm=load_llm(HUGGINGFACE_REPO_ID),
#     chain_type="stuff",
#     retriever=db.as_retriever(search_kwargs={'k': 5}),
#     return_source_documents=True,
#     chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
# )

# @csrf_exempt
# def chatbot_response(request):
#     if request.method == 'POST':
#         try:
#             data = json.loads(request.body)
#             prompt = data.get('prompt')

#             response = qa_chain.invoke({'query': prompt})
#             source_docs = response.get('source_documents', [])

#             if not source_docs:
#                 return JsonResponse({'response': "I don't know the answer."})

#             # Print page numbers from source docs
#             print("Source Pages:")
#             for doc in source_docs:
#                 if 'page' in doc.metadata:
#                     print(f"Page: {doc.metadata['page']}")
#                 else:
#                     print("Page information not found.")

#             result = response.get('result', "I don't know the answer.")
#             return JsonResponse({'response': result})

#         except Exception as e:
#             return JsonResponse({'error': str(e)}, status=500)
#     elif request.method == 'GET':
#         return render(request, 'chatbot/index.html')
#     return JsonResponse({'error': 'Invalid request method'}, status=400)



# for auth ====

# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# from django.shortcuts import render, redirect
# from django.contrib.auth import authenticate, login, logout
# from django.contrib.auth.decorators import login_required
# from .models import CustomUser
# import json
# from langchain_huggingface import HuggingFaceEndpoint
# from langchain_core.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# import os
# from dotenv import load_dotenv, find_dotenv

# load_dotenv(find_dotenv())

# HF_TOKEN = os.environ.get("HF_TOKEN")
# HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# def load_llm(huggingface_repo_id):
#     llm = HuggingFaceEndpoint(
#         repo_id=huggingface_repo_id,
#         temperature=0.5,
#         model_kwargs={"use_auth_token": HF_TOKEN, "max_length": 512},
#         task="text-generation"
#     )
#     return llm

# CUSTOM_PROMPT_TEMPLATE = """
# Use the pieces of information provided in the context to answer the user's question.
# If you don't know the answer, just say that you don't know. Don't try to make up an answer. 
# Don't provide anything out of the given context.

# Context: {context}
# Question: {question}

# Start the answer directly. No small talk, please.And Answer in a way that anybody can understand , even uneducated/ Non-medical background person can also understand it.
# """

# def set_custom_prompt(custom_prompt_template):
#     return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

# DB_FAISS_PATH = "vectorstore/db_faiss"
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# qa_chain = RetrievalQA.from_chain_type(
#     llm=load_llm(HUGGINGFACE_REPO_ID),
#     chain_type="stuff",
#     retriever=db.as_retriever(search_kwargs={'k': 5}),
#     return_source_documents=True,
#     chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
# )

# def signup(request):
#     if request.method == 'POST':
#         username = request.POST.get('username')
#         password = request.POST.get('password')
#         user_type = request.POST.get('user_type')
#         if CustomUser.objects.filter(username=username).exists():
#             return render(request, 'chatbot/signup.html', {'error': 'Username already exists'})
#         user = CustomUser.objects.create_user(username=username, password=password, user_type=user_type)
#         login(request, user)
#         return redirect('chatbot_response')
#     return render(request, 'chatbot/signup.html')

# def user_login(request):
#     if request.method == 'POST':
#         username = request.POST.get('username')
#         password = request.POST.get('password')
#         user = authenticate(request, username=username, password=password)
#         if user is not None:
#             login(request, user)
#             return redirect('chatbot_response')
#         return render(request, 'chatbot/login.html', {'error': 'Invalid credentials'})
#     return render(request, 'chatbot/login.html')

# def user_logout(request):
#     logout(request)
#     return redirect('user_login')

# @csrf_exempt
# @login_required
# def chatbot_response(request):
#     if request.method == 'POST':
#         try:
#             data = json.loads(request.body)
#             prompt = data.get('prompt')

#             response = qa_chain.invoke({'query': prompt})
#             source_docs = response.get('source_documents', [])

#             if not source_docs:
#                 return JsonResponse({'response': "I don't know the answer."})

#             # Print page numbers from source docs
#             print("Source Pages:")
#             for doc in source_docs:
#                 if 'page' in doc.metadata:
#                     print(f"Page: {doc.metadata['page']}")
#                 else:
#                     print("Page information not found.")

#             result = response.get('result', "I don't know the answer.")
#             return JsonResponse({'response': result})

#         except Exception as e:
#             return JsonResponse({'error': str(e)}, status=500)
#     elif request.method == 'GET':
#         return render(request, 'chatbot/index.html')
#     return JsonResponse({'error': 'Invalid request method'}, status=400)



# views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from .models import CustomUser
import json
import os
import tempfile
from dotenv import load_dotenv, find_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from .brain_of_doctor import analyze_image_with_query, encode_image

# Load environment variables
load_dotenv(find_dotenv())
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Load the language model using Groq
def load_llm_groq():
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="mistral-saba-24b",  # You can also use llama3-8b-8192
        temperature=0.5
    )
    return llm

# Custom prompt for understandable answers
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know. Don't try to make up an answer. 
Don't provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk, please. And answer in a way that anybody can understand, even someone with no medical or technical background.
"""

def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

# Load FAISS vector store
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QA chain using Groq
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm_groq(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 8}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Authentication Views
def signup(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm_password')
        user_type = request.POST.get('user_type')
        full_name=request.POST.get('full_name')
        # Check if passwords match
        if password != confirm_password:
            return render(request, 'chatbot/signup.html', {'error': 'Passwords do not match'})
            
        if CustomUser.objects.filter(username=username).exists():
            return render(request, 'chatbot/signup.html', {'error': 'Username already exists'})
        # Create the user
        user = CustomUser.objects.create_user(username=username, password=password, user_type=user_type)

        if 'profile_picture' in request.FILES:
            user.profile_picture = request.FILES['profile_picture']
        
        # Save doctor-specific fields if applicable
        if user_type == 'doctor':
            user.doctor_id = request.POST.get('doctor_id')
            user.specialization = request.POST.get('specialization')
            user.experience = request.POST.get('experience')
            user.clinic_address = request.POST.get('clinic_address')
            user.contact_number = request.POST.get('contact_number')
            user.full_name = request.POST.get('full_name')
            user.profile_picture = request.FILES.get('profile_picture')
            user.save()
        
        # Redirect to login instead of auto-login
        return redirect('user_login')
    
    return render(request, 'chatbot/signup.html')


def user_login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        login_user_type = request.POST.get('login_user_type')
        
        # Additional field for doctors
        doctor_id = request.POST.get('doctor_id', None) if login_user_type == 'doctor' else None
        
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            # Check if user type matches
            if user.user_type == login_user_type:
                # Additional verification for doctors
                if login_user_type == 'doctor':
                    if not doctor_id:
                        return render(request, 'chatbot/login.html', {
                            'error': 'Doctor ID is required',
                            'show_doctor_id': True  # This will show the doctor ID field in template
                        })
                    if user.doctor_id != doctor_id:
                        return render(request, 'chatbot/login.html', {
                            'error': 'Invalid Doctor ID',
                            'show_doctor_id': True
                        })
                
                login(request, user)
                
                # Redirect to appropriate dashboard based on user type
                if login_user_type == 'doctor':
                    return redirect('doctor_dashboard')
                else:
                    return redirect('dashboard')
            else:
                return render(request, 'chatbot/login.html', {
                    'error': f'Invalid login. This user is not registered as a {login_user_type}',
                    'show_doctor_id': (login_user_type == 'doctor')  # Show field if doctor was selected
                })
        else:
            return render(request, 'chatbot/login.html', {
                'error': 'Invalid credentials',
                'show_doctor_id': (login_user_type == 'doctor')  # Show field if doctor was selected
            })
    
    return render(request, 'chatbot/login.html', {'show_doctor_id': False})

def user_logout(request):
    logout(request)
    return redirect('user_login')

# Page Views
@login_required
def medical_assistant(request):
    return render(request, 'chatbot/index.html')

@login_required
def dashboard_view(request):
    upcoming_count = 0
    if request.user.is_authenticated and request.user.is_patient():
        upcoming_count = Appointment.objects.filter(
            patient=request.user,
            date__gte=timezone.now().date()
        ).count()
    
    return render(request, 'chatbot/dashboard.html', {
        'upcoming_count': upcoming_count
    })

@login_required
def doctor_dashboard_view(request):
    if request.user.user_type != 'doctor':
        return redirect('dashboard')
    
    # Get upcoming appointments for this doctor
    appointments = Appointment.objects.filter(
        doctor=request.user,
        date__gte=timezone.now().date()
    ).order_by('date', 'time')
    
    return render(request, 'chatbot/doctor_dashboard.html', {
        'appointments': appointments
    })

from django.contrib import messages
from django.shortcuts import get_object_or_404

@login_required
def create_prescription(request):
    if request.method == 'POST':
        if request.user.user_type != 'doctor':
            messages.error(request, 'Only doctors can create prescriptions.')
            return redirect('doctor_dashboard')
        
        appointment_id = request.POST.get('appointment_id')
        appointment = get_object_or_404(Appointment, id=appointment_id, doctor=request.user)
        
        Prescription.objects.create(
            appointment=appointment,
            doctor=request.user,
            patient=appointment.patient,
            medication=request.POST.get('medication'),
            dosage=request.POST.get('dosage'),
            instructions=request.POST.get('instructions'),
            duration=request.POST.get('duration')
        )
        
        messages.success(request, 'Prescription created successfully!')
        return redirect('doctor_dashboard')
    
    return redirect('doctor_dashboard')
# Chatbot Logic
@csrf_exempt
@login_required
def chatbot_response(request):
    if request.method == 'POST':
        try:
            # Handle image-based queries
            if request.FILES:
                files = request.FILES.getlist('files')
                prompt = request.POST.get('prompt', '')

                if not files:
                    return JsonResponse({'response': 'Please upload an image for analysis'})

                for file in files:
                    if file.content_type.startswith('image/'):
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                            for chunk in file.chunks():
                                tmp.write(chunk)
                            tmp_path = tmp.name

                        try:
                            encoded_image = encode_image(tmp_path)
                            response = analyze_image_with_query(
                                prompt if prompt else "Is there something wrong in this image?",
                                "meta-llama/llama-4-scout-17b-16e-instruct",  # image model if needed
                                encoded_image
                            )
                            return JsonResponse({'response': response})
                        finally:
                            os.unlink(tmp_path)

                return JsonResponse({'response': 'No valid image file was uploaded'})

            # Handle text-based queries
            else:
                data = json.loads(request.body)
                prompt = data.get('prompt')

                chain_input = {'query': prompt}
                print("Prompt input to QA Chain:", prompt)
                print("Chain input dict:", chain_input)

                response = qa_chain.invoke(chain_input)

                result = response.get("result", "").strip()
                source_docs = response.get("source_documents", [])
                print("QA Chain Output Result:", result)
                print("Number of Source Docs:", len(source_docs))


                result = response.get("result", "").strip()
                source_docs = response.get("source_documents", [])

                if not source_docs or result == "":
                    return JsonResponse({'response': "I don't know the answer."})

                return JsonResponse({'response': result})

        except Exception as e:
            import traceback
            print("Exception occurred:", str(e))
            traceback.print_exc()
            return JsonResponse({'error': str(e)}, status=500)


    elif request.method == 'GET':
        return render(request, 'chatbot/index.html')

    return JsonResponse({'error': 'Invalid request method'}, status=400)


from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.utils import timezone
from .models import CustomUser, Appointment

@login_required
def appointment_view(request):
    if not request.user.is_patient():
        return redirect('dashboard')
    
    # Get upcoming appointments
    appointments = Appointment.objects.filter(
        patient=request.user,
        date__gte=timezone.now().date()
    ).order_by('date', 'time')
    
    return render(request, 'chatbot/appointments.html', {
        'appointments': appointments,
        'upcoming_count': appointments.count()
    })

@login_required
def find_doctor_view(request):
    if not request.user.is_patient():
        return redirect('dashboard')
    
    doctors = CustomUser.objects.filter(user_type='doctor')
    specialties = CustomUser.objects.filter(
        user_type='doctor',
        specialization__isnull=False
    ).values_list('specialization', flat=True).distinct()
    
    # Get search parameters
    specialty = request.GET.get('specialty', '')
    location = request.GET.get('location', '')
    
    if specialty:
        doctors = doctors.filter(specialization__icontains=specialty)
    if location:
        doctors = doctors.filter(clinic_address__icontains=location)
    
    return render(request, 'chatbot/find_doctors.html', {
        'doctors': doctors,
        'specialties': specialties,
        'selected_specialty': specialty,
        'selected_location': location
    })

@login_required
def book_appointment_view(request, doctor_id):
    if not request.user.is_patient():
        return redirect('dashboard')
    
    doctor = get_object_or_404(CustomUser, id=doctor_id, user_type='doctor')
    
    if request.method == 'POST':
        date = request.POST.get('date')
        time = request.POST.get('time')
        reason = request.POST.get('reason')
        
        # Validate date is not in the past
        if timezone.now().date() > timezone.datetime.strptime(date, '%Y-%m-%d').date():
            messages.error(request, "You can't book an appointment in the past")
            return redirect('book_appointment', doctor_id=doctor_id)
        
        # Create the appointment
        Appointment.objects.create(
            patient=request.user,
            doctor=doctor,
            date=date,
            time=time,
            reason=reason,
            status='pending'
        )
        
        messages.success(request, "Appointment booked successfully!")
        return redirect('appointments')
    
    return render(request, 'chatbot/book_appointment.html', {'doctor': doctor})

@login_required
def manage_appointments_view(request):
    if request.user.is_doctor():
        appointments = Appointment.objects.filter(
            doctor=request.user,
            date__gte=timezone.now().date()
        ).order_by('date', 'time')
    else:
        appointments = Appointment.objects.filter(
            patient=request.user,
            date__gte=timezone.now().date()
        ).order_by('date', 'time')
    
    return render(request, 'chatbot/manage_appointments.html', {
        'appointments': appointments,
        'is_doctor': request.user.is_doctor()
    })


from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .models import Medication, CommonMedication
from django.utils import timezone

@login_required
def medication_view(request):
    # Get current medications for the user
    current_meds = Medication.objects.filter(
        patient=request.user,
        is_active=True
    ).order_by('-start_date')

    # Get common medications
    common_meds = CommonMedication.objects.all()

    if request.method == 'POST':
        # Handle form submission
        name = request.POST.get('name')
        purpose = request.POST.get('purpose', '')
        dosage = request.POST.get('dosage')
        frequency = int(request.POST.get('frequency'))
        start_date = request.POST.get('start_date')
        end_date = request.POST.get('end_date') or None
        notes = request.POST.get('notes', '')

        Medication.objects.create(
            patient=request.user,
            name=name,
            purpose=purpose,
            dosage=dosage,
            frequency=frequency,
            start_date=start_date,
            end_date=end_date,
            notes=notes
        )
        return redirect('medications')

    context = {
        'current_meds': current_meds,
        'common_meds': common_meds,
        'today': timezone.now().date()
    }
    return render(request, 'chatbot/medications.html', context)

@login_required
def medication_toggle(request, med_id):
    if request.method == 'POST':
        medication = get_object_or_404(Medication, id=med_id, patient=request.user)
        medication.is_active = not medication.is_active
        medication.save()
    return redirect('medications')

@login_required
def medication_delete(request, med_id):
    if request.method == 'POST':
        medication = get_object_or_404(Medication, id=med_id, patient=request.user)
        medication.delete()
    return redirect('medications')


from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from .models import Medication, Prescription, MedicalRecord, CustomUser
from django.utils import timezone

@login_required
def medical_records_view(request):
    # Get all medications added by the patient
    medications = Medication.objects.filter(patient=request.user, is_active=True)
    
    # Get all prescriptions for the patient
    prescriptions = Prescription.objects.filter(patient=request.user, is_active=True)
    
    # Get all medical records for the patient
    medical_records = MedicalRecord.objects.filter(patient=request.user)
    
    # Get list of doctors for the upload form
    doctors = CustomUser.objects.filter(user_type='doctor')
    
    context = {
        'medications': medications,
        'prescriptions': prescriptions,
        'medical_records': medical_records,
        'doctors': doctors,
        'today': timezone.now().date()
    }
    return render(request, 'chatbot/medical_records.html', context)

@login_required
def upload_medical_record(request):
    if request.method == 'POST':
        record_type = request.POST.get('record_type')
        title = request.POST.get('title')
        date = request.POST.get('date')
        doctor_id = request.POST.get('doctor')
        file = request.FILES.get('file')
        notes = request.POST.get('notes')
        
        # Validate file type
        allowed_extensions = ['.pdf', '.jpg', '.jpeg', '.png']
        file_extension = os.path.splitext(file.name)[1].lower()
        if file_extension not in allowed_extensions:
            messages.error(request, 'Invalid file type. Please upload PDF, JPG, JPEG, or PNG files.')
            return redirect('medical_records')
        
        # Save the file
        fs = FileSystemStorage()
        filename = fs.save(f'medical_records/user_{request.user.id}/{file.name}', file)
        file_url = fs.url(filename)
        
        # Get doctor instance if selected
        doctor = None
        if doctor_id:
            doctor = CustomUser.objects.get(id=doctor_id)
        
        # Create medical record
        MedicalRecord.objects.create(
            patient=request.user,
            record_type=record_type,
            title=title,
            date=date,
            doctor=doctor,
            file=filename,
            notes=notes
        )
        
        messages.success(request, 'Medical record uploaded successfully!')
        return redirect('medical_records')
    
    return redirect('medical_records')



# views.py
from django.shortcuts import render, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from .models import HealthTip, HealthTipCategory, SavedTip, WellnessArticle

def health_tips_view(request):
    categories = HealthTipCategory.objects.all()
    selected_category = request.GET.get('category', 'all')
    
    if selected_category != 'all':
        health_tips = HealthTip.objects.filter(category__name=selected_category)
    else:
        health_tips = HealthTip.objects.all()
    
    wellness_articles = WellnessArticle.objects.order_by('-published_date')[:5]
    
    # Check which tips are saved by the user
    saved_tips = []
    if request.user.is_authenticated:
        saved_tips = SavedTip.objects.filter(user=request.user).values_list('tip_id', flat=True)
    
    return render(request, 'chatbot/health_tips.html', {
        'health_tips': health_tips,
        'wellness_articles': wellness_articles,
        'categories': categories,
        'selected_category': selected_category,
        'saved_tips': saved_tips
    })

@login_required
def save_tip(request, tip_id):
    tip = get_object_or_404(HealthTip, id=tip_id)
    saved_tip, created = SavedTip.objects.get_or_create(user=request.user, tip=tip)
    
    if created:
        return JsonResponse({'status': 'saved', 'message': 'Tip saved successfully'})
    else:
        saved_tip.delete()
        return JsonResponse({'status': 'removed', 'message': 'Tip removed from saved'})

@login_required
def saved_tips_view(request):
    saved_tips = SavedTip.objects.filter(user=request.user).select_related('tip')
    return render(request, 'chatbot/saved_tips.html', {'saved_tips': saved_tips})







# -------working ---------
# from django.http import JsonResponse 
# from django.views.decorators.csrf import csrf_exempt
# from django.shortcuts import render
# import json
# import os
# from dotenv import load_dotenv, find_dotenv
# from langchain_huggingface import HuggingFaceEndpoint
# from langchain_core.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS

# load_dotenv(find_dotenv())

# # Load Hugging Face Token and Model Repo
# HF_TOKEN = os.environ.get("HF_TOKEN")
# HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# # Load Language Model from Hugging Face
# def load_llm(huggingface_repo_id):
#     llm = HuggingFaceEndpoint(
#         repo_id=huggingface_repo_id,
#         temperature=0.5,
#         model_kwargs={
#             "use_auth_token": HF_TOKEN,
#             "max_length": 512
#         },
#         task="text-generation"
#     )
#     return llm

# # Custom Prompt Template
# CUSTOM_PROMPT_TEMPLATE = """
# Use the pieces of information provided in the context to answer the user's question.
# If you don't know the answer, just say that you don't know. Don't try to make up an answer. 
# Don't provide anything out of the given context.

# Context: {context}
# Question: {question}

# Start the answer directly. No small talk, please.
# """

# def set_custom_prompt(custom_prompt_template):
#     prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
#     return prompt

# # Load FAISS Vector Database
# DB_FAISS_PATH = "vectorstore/db_faiss"
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# # Create RetrievalQA Chain
# qa_chain = RetrievalQA.from_chain_type(
#     llm=load_llm(HUGGINGFACE_REPO_ID),
#     chain_type="stuff",
#     retriever=db.as_retriever(search_kwargs={'k': 5}),
#     return_source_documents=True,
#     chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
# )

# @csrf_exempt
# def chatbot_response(request):
#     if request.method == 'POST':
#         try:
#             data = json.loads(request.body)
#             prompt = data.get('prompt')
            
#             # Process the user input using the QA chain
#             response = qa_chain.invoke({'query': prompt})
#             result = response["result"]
            
#             return JsonResponse({'response': result})
#         except Exception as e:
#             return JsonResponse({'error': str(e)}, status=500)
#     else:
#         return render(request, 'chatbot/index.html')
