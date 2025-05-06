# import streamlit as st

# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA

# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import PromptTemplate
# from langchain_huggingface import HuggingFaceEndpoint

# DB_FAISS_PATH="vectorstore/db_faiss"
# @st.cache_resource
# def get_vectorstore():
#     embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
#     db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
#     return db

# def set_custom_prompt(custom_prompt_template):
#     prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
#     return prompt

# def load_llm(huggingface_repo_id, HF_TOKEN):
#     llm=HuggingFaceEndpoint(
#         repo_id=huggingface_repo_id,
#         temperature=0.5,
#         model_kwargs={"token":HF_TOKEN,
#                       "max_length":"512"}
#     )
#     return llm

# def main():
#   st.title("Ask Chatbot!")

#   if 'messages' not in st.session_state:
#         st.session_state.messages = []

#   for message in st.session_state.messages:
#     st.chat_message(message['role']).markdown(message['content'])

  
#   prompt = st.chat_input("Pass your prompt here")

#   if prompt:
#     st.chat_message('user').markdown(prompt)
#     st.session_state.messages.append({'role':'user', 'content': prompt})

#     CUSTOM_PROMPT_TEMPLATE = """
#                 Use the pieces of information provided in the context to answer user's question.
#                 If you dont know the answer, just say that you dont know, dont try to make up an answer. 
#                 Dont provide anything out of the given context

#                 Context: {context}
#                 Question: {question}

#                 Start the answer directly. No small talk please.
#                 """

#     response = "Hi, I am MediCare Bot v1.0"
#     st.chat_message('assistant').markdown(response)
#     st.session_state.messages.append({'role':'assistant', 'content': response})

# if __name__ == "__main__":
#   main()

# import os
# import streamlit as st

# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA

# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import PromptTemplate
# from langchain_huggingface import HuggingFaceEndpoint

# ## Uncomment the following files if you're not using pipenv as your virtual environment manager
# #from dotenv import load_dotenv, find_dotenv
# #load_dotenv(find_dotenv())


# DB_FAISS_PATH="vectorstore/db_faiss"
# @st.cache_resource
# def get_vectorstore():
#     embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
#     db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
#     return db


# def set_custom_prompt(custom_prompt_template):
#     prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
#     return prompt


# def load_llm(huggingface_repo_id, HF_TOKEN):
#     llm=HuggingFaceEndpoint(
#         repo_id=huggingface_repo_id,
#         temperature=0.5,
#         model_kwargs={"token":HF_TOKEN,
#                       "max_length":"512"},
#                       task="text-generation"
#     )
#     return llm


# def main():
#     st.title("Ask Chatbot!")

#     if 'messages' not in st.session_state:
#         st.session_state.messages = []

#     for message in st.session_state.messages:
#         st.chat_message(message['role']).markdown(message['content'])

#     prompt=st.chat_input("Pass your prompt here")

#     if prompt:
#         st.chat_message('user').markdown(prompt)
#         st.session_state.messages.append({'role':'user', 'content': prompt})

#         CUSTOM_PROMPT_TEMPLATE = """
#                 Use the pieces of information provided in the context to answer user's question.
#                 If you dont know the answer, just say that Sorry! you dont know, dont try to make up an answer. 
#                 Dont provide anything out of the given context

#                 Context: {context}
#                 Question: {question}

#                 Start the answer directly. No small talk please.
#                 """
        
#         HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"
#         HF_TOKEN=os.environ.get("HF_TOKEN")

#         try: 
#             vectorstore=get_vectorstore()
#             if vectorstore is None:
#                 st.error("Failed to load the vector store")

#             qa_chain=RetrievalQA.from_chain_type(
#                 llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
#                 chain_type="stuff",
#                 retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
#                 return_source_documents=True,
#                 chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
#             )

#             response=qa_chain.invoke({'query':prompt})

#             result=response["result"]
#             source_documents=response["source_documents"]
#             # result_to_show=result+"\nSource Docs:\n"+str(source_documents)
#             result_to_show=result
#             #response="Hi, I am MediBot!"
#             st.chat_message('assistant').markdown(result_to_show)
#             st.session_state.messages.append({'role':'assistant', 'content': result_to_show})

#         except Exception as e:
#             st.error(f"Error: {str(e)}")

# if __name__ == "__main__":
#     main()


# import os
# import streamlit as st
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import PromptTemplate
# from langchain_huggingface import HuggingFaceEndpoint

# # Add custom CSS for styling
# def set_custom_style():
#     st.markdown("""
#     <style>
#         /* Main container styling */
#         .main {
#             background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
#             padding: 2rem;
#         }
        
#         /* Header styling */
#         .header {
#             padding: 1rem;
#             border-radius: 15px;
#             background: rgba(255, 255, 255, 0.9);
#             box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
#             margin-bottom: 2rem;
#         }
        
#         /* Chat message styling */
#         .user-message {
#             background: #ffffff;
#             border-radius: 15px 15px 0 15px;
#             padding: 1rem;
#             margin: 0.5rem 0;
#             box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
#         }
        
#         .assistant-message {
#             background: #e3f2fd;
#             border-radius: 15px 15px 15px 0;
#             padding: 1rem;
#             margin: 0.5rem 0;
#             box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
#         }
        
#         /* Input box styling */
#         .stChatInput {
#             margin-top: 2rem;
#             border-radius: 20px;
#             box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
#         }
        
#         /* Title styling */
#         .title-text {
#             color: #1e3a8a;
#             font-family: 'Arial Rounded MT Bold', sans-serif;
#             font-size: 2.5rem !important;
#         }
        
#         /* Side decorations */
#         .decoration-bar {
#             height: 5px;
#             background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
#             margin: 1rem 0;
#             border-radius: 2px;
#         }
#     </style>
#     """, unsafe_allow_html=True)

# DB_FAISS_PATH = "vectorstore/db_faiss"

# @st.cache_resource
# def get_vectorstore():
#     embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
#     db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
#     return db

# def set_custom_prompt(custom_prompt_template):
#     prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
#     return prompt

# def load_llm(huggingface_repo_id, HF_TOKEN):
#     llm = HuggingFaceEndpoint(
#         repo_id=huggingface_repo_id,
#         temperature=0.5,
#         model_kwargs={"token": HF_TOKEN, "max_length": "512"},
#         task="text-generation"
#     )
#     return llm

# def main():
#     set_custom_style()
    
#     # Custom header with icon
#     col1, col2 = st.columns([0.1, 0.9])
#     with col1:
#         st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=80)
#     with col2:
#         st.markdown("<div class='header'><h1 class='title-text'>MediCareBot Assistant</h1><div class='decoration-bar'></div><p>Your intelligent healthcare companion 🤖💊</p></div>", unsafe_allow_html=True)

#     if 'messages' not in st.session_state:
#         st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm MediCare Bot. How can I assist you with healthcare information today? 🩺"}]

#     # Display chat messages
#     for message in st.session_state.messages:
#         if message["role"] == "user":
#             st.markdown(f"""
#             <div style="display: flex; align-items: center; margin-bottom: 1rem;">
#                 <div style="margin-right: 1rem;">👤</div>
#                 <div class="user-message">{message['content']}</div>
#             </div>
#             """, unsafe_allow_html=True)
#         else:
#             st.markdown(f"""
#             <div style="display: flex; align-items: center; margin-bottom: 1rem;">
#                 <div style="margin-right: 1rem;">🤖</div>
#                 <div class="assistant-message">{message['content']}</div>
#             </div>
#             """, unsafe_allow_html=True)

#     # Chat input
#     prompt = st.chat_input("Type your health-related question here...")
    
#     if prompt:
#         # Display user message
#         st.markdown(f"""
#         <div style="display: flex; align-items: center; margin-bottom: 1rem;">
#             <div style="margin-right: 1rem;">👤</div>
#             <div class="user-message">{prompt}</div>
#         </div>
#         """, unsafe_allow_html=True)
#         st.session_state.messages.append({'role': 'user', 'content': prompt})

#         # Process query
#         CUSTOM_PROMPT_TEMPLATE = """
#         Use the pieces of information provided in the context to answer user's question.
#         If you don't know the answer, just say that Sorry! you don't know.Give answer in pointwise whenever necessary.
#         Don't provide anything out of the given context

#         Context: {context}
#         Question: {question}

#         """
        
#         HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
#         HF_TOKEN = os.environ.get("HF_TOKEN")

#         try: 
#             with st.spinner('🔍 Searching for the best answer...'):
#                 vectorstore = get_vectorstore()
#                 qa_chain = RetrievalQA.from_chain_type(
#                     llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
#                     chain_type="stuff",
#                     retriever=vectorstore.as_retriever(search_kwargs={'k': 6}),
#                     return_source_documents=True,
#                     chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
#                 )

#                 response = qa_chain.invoke({'query': prompt})
#                 result = response["result"]
                
#                 # Display assistant response
#                 st.markdown(f"""
#                 <div style="display: flex; align-items: center; margin-bottom: 1rem;">
#                     <div style="margin-right: 1rem;">🤖</div>
#                     <div class="assistant-message">{result}</div>
#                 </div>
#                 """, unsafe_allow_html=True)
#                 st.session_state.messages.append({'role': 'assistant', 'content': result})

#         except Exception as e:
#             st.error(f"🚨 Error: {str(e)}")

# if __name__ == "__main__":
#     main()


# -------1st Main-------

import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from streamlit_mic_recorder import mic_recorder
from gtts import gTTS
import speech_recognition as sr
import base64
from pydub import AudioSegment
import io

# Set the path to the FFmpeg executable
ffmpeg_path = "C:/Users/Chetan/Downloads/ffmpeg-master-latest-win64-gpl-shared/ffmpeg-master-latest-win64-gpl-shared/bin/ffmpeg.exe"  # Update this path

ffprobe_path = "c:/Users/Chetan/Downloads/ffmpeg-master-latest-win64-gpl-shared/ffmpeg-master-latest-win64-gpl-shared/bin/ffprobe.exe"  # Update this path

AudioSegment.ffmpeg = ffmpeg_path
AudioSegment.ffprobe = ffprobe_path

# Add custom CSS for styling and animations
def set_custom_style():
    st.markdown("""
    <style>
        /* Main container styling */
        .main {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 2rem;
        }
        
        /* Header styling */
        .header {
            padding: 1rem;
            border-radius: 15px;
            background: rgba(255, 255, 255, 0.9);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
        }
        
        /* Chat message styling */
        .user-message {
            background: #ffffff;
            border-radius: 15px 15px 0 15px;
            padding: 1rem;
            margin: 0.5rem 0;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            animation: slideInLeft 0.5s ease-out;
        }
        
        .assistant-message {
            background: #e3f2fd;
            border-radius: 15px 15px 15px 0;
            padding: 1rem;
            margin: 0.5rem 0;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            animation: slideInRight 0.5s ease-out;
        }
        
        /* Input box styling */
        .stChatInput {
            margin-top: 2rem;
            border-radius: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        /* Title styling */
        .title-text {
            color: #1e3a8a;
            font-family: 'Arial Rounded MT Bold', sans-serif;
            font-size: 2.5rem !important;
        }
        
        /* Side decorations */
        .decoration-bar {
            height: 5px;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            margin: 1rem 0;
            border-radius: 2px;
        }
        
        /* Animations */
        @keyframes slideInLeft {
            from { transform: translateX(-100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        
        @keyframes slideInRight {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        
        /* Quick action buttons */
        .quick-action-btn {
            margin: 0.5rem;
            padding: 0.75rem 1.5rem;
            border-radius: 25px;
            background: #667eea;
            color: white;
            border: none;
            cursor: pointer;
            transition: background 0.3s ease;
        }
        
        .quick-action-btn:hover {
            background: #764ba2;
        }
    </style>
    """, unsafe_allow_html=True)

# Function to convert audio to WAV format
def convert_audio_to_wav(audio_bytes):
    try:
        # Load audio using pydub (assumes the input is in WebM/Opus format)
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="webm")
        
        # Export to WAV format
        wav_audio = io.BytesIO()
        audio.export(wav_audio, format="wav")
        wav_audio.seek(0)
        return wav_audio
    except Exception as e:
        st.error(f"Error converting audio to WAV: {str(e)}")
        return None

# Function to convert speech to text
def speech_to_text(audio_bytes):
    recognizer = sr.Recognizer()
    
    # Convert audio to WAV format
    wav_audio = convert_audio_to_wav(audio_bytes)
    if wav_audio is None:
        return "Could not process audio"
    
    # Use the WAV audio with speech_recognition
    with sr.AudioFile(wav_audio) as source:
        audio = recognizer.record(source)
    
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError:
        return "Could not request results"

# Function to convert text to speech
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    tts.save("response.mp3")
    with open("response.mp3", "rb") as audio_file:
        audio_bytes = audio_file.read()
    return audio_bytes

# Function to autoplay audio
def autoplay_audio(audio_bytes):
    b64 = base64.b64encode(audio_bytes).decode()
    md = f"""
    <audio controls autoplay>
    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """
    st.markdown(md, unsafe_allow_html=True)

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm(huggingface_repo_id, HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"},
        task="text-generation"
    )
    return llm

def main():
    set_custom_style()
    
    # Sidebar for additional information
    with st.sidebar:
        st.markdown("## 📌 Quick Actions")
        if st.button("What are common symptoms of flu?"):
            st.session_state.prompt = "What are common symptoms of flu?"
        if st.button("How to manage diabetes?"):
            st.session_state.prompt = "How to manage diabetes?"
        if st.button("What is a healthy diet?"):
            st.session_state.prompt = "What is a healthy diet?"
        if st.button("How to reduce stress?"):
            st.session_state.prompt = "How to reduce stress?"
        
        st.markdown("---")
        st.markdown("## 🎙️ Voice Input")
        audio = mic_recorder(start_prompt="🎤 Start Recording", stop_prompt="⏹️ Stop Recording", key="recorder")
        
        if audio:
            # Convert the recorded audio to text
            text = speech_to_text(audio['bytes'])
            if text:
                st.session_state.prompt = text  # Auto-fill the input field
        
        st.markdown("---")
        st.markdown("## ℹ️ About MediBot")
        st.markdown("MediBot is your intelligent healthcare assistant. It provides reliable and accurate health-related information based on trusted sources.")
        st.markdown("**Disclaimer:** This is not a substitute for professional medical advice.")

    # Custom header with icon
    col1, col2 = st.columns([0.1, 0.9])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=80)
    with col2:
        st.markdown("<div class='header'><h1 class='title-text'>MediBot Assistant</h1><div class='decoration-bar'></div><p>Your intelligent healthcare companion 🤖💊</p></div>", unsafe_allow_html=True)

    if 'messages' not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm MediBot. How can I assist you with healthcare information today? 🩺"}]

    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <div style="margin-right: 1rem;">👤</div>
                <div class="user-message">{message['content']}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <div style="margin-right: 1rem;">🤖</div>
                <div class="assistant-message">{message['content']}</div>
            </div>
            """, unsafe_allow_html=True)

    # Chat input
    prompt = st.chat_input("Type your health-related question here...")
    
    # Check if a quick action button or voice input was used
    if 'prompt' in st.session_state:
        prompt = st.session_state.prompt
        del st.session_state.prompt
    
    if prompt:
        # Display user message
        st.markdown(f"""
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <div style="margin-right: 1rem;">👤</div>
            <div class="user-message">{prompt}</div>
        </div>
        """, unsafe_allow_html=True)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        # Process query
        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer user's question.
        If you don't know the answer, just say that Sorry! I don't know, Give answer in pointwise whenever necessary. 
        Don't provide anything out of the given context

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk please.
        """
        
        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")

        try: 
            with st.spinner('🔍 Searching for the best answer...'):
                vectorstore = get_vectorstore()
                qa_chain = RetrievalQA.from_chain_type(
                    llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                    return_source_documents=True,
                    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
                )

                response = qa_chain.invoke({'query': prompt})
                result = response["result"]
                
                # Display assistant response
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                    <div style="margin-right: 1rem;">🤖</div>
                    <div class="assistant-message">{result}</div>
                </div>
                """, unsafe_allow_html=True)
                st.session_state.messages.append({'role': 'assistant', 'content': result})

                # Convert response to speech
                audio_bytes = text_to_speech(result)
                st.audio(audio_bytes, format="audio/mp3")

        except Exception as e:
            st.error(f"🚨 Error: {str(e)}")

if __name__ == "__main__":
    main()



