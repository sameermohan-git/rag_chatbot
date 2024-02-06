import streamlit as st
import pdfplumber
import os
import random
#from question_generator import generate_questions
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from streamlit_chat import message

import pinecone
from langchain_community.vectorstores import Pinecone

#import utils
import openai
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import base64

os.environ['KMP_DUPLICATE_LIB_OK']='True'

_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key  = os.getenv('OPENAI_API_KEY')

# Initialize LangChain LLM and Memory
llm_model = "gpt-3.5-turbo-0613"
llm = ChatOpenAI(temperature=0.0, model=llm_model)
 
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_INDEX_NAME="rag-level5-idx"
pinecone_environment = 'gcp-starter' 

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)     
index = pc.Index(PINECONE_INDEX_NAME)

# Initial page config

st.set_page_config(
     page_title='Ask me anything about your data!',
     layout="wide",
     initial_sidebar_state="expanded",
)

def main():
    #st.title("Ask a PDF Questions")
    cs_sidebar()
    cs_body()


##########################
# Main body of Chat bot
##########################

def cs_body():
    st.title("💬 Level5 Strategy Chatbot")
    st.caption("🚀 aisemble chatbot powered by OpenAI LLM")
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""
    
    if 'retriever' not in st.session_state:
        with st.spinner("Loading existing data..."):
            index_name = PINECONE_INDEX_NAME
            embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
            docsearch = Pinecone.from_existing_index(index_name, embeddings)
            st.session_state.retriever = docsearch.as_retriever()
        #st.info("PDF already processed. Using existing data.")
        st.session_state.pdf_processed = True
    
    if st.session_state.pdf_processed:
        for idx, (speaker, text) in enumerate(st.session_state.chat_history):
            if speaker == "Bot":
                message(text, key=f"msg-{idx}")
            else:
                message(text, is_user=True, key=f"msg-{idx}")

        st.text_input("How can I help you?", key="user_input", on_change=handle_enter, placeholder="Your AI assistant here! Ask me anything")

        if st.session_state.user_input:
            handle_enter()

    return None

# sidebar
def cs_sidebar():

    st.sidebar.markdown('''[<img src='data:image/png;base64,{}' class='img-fluid' width=32 height=32>](https://level5strategy.com/)'''.format(img_to_bytes("images/L5Logo.png")), unsafe_allow_html=True)
    st.sidebar.header('Upload Pdf')

    uploaded_file = st.sidebar.file_uploader("Upload your PDF here", type="pdf")
    if uploaded_file:
        file_name = uploaded_file.name
        if not has_been_processed(file_name):
            with st.sidebar.spinner("Processing PDF..."):
                pages = extract_text_from_pdf(uploaded_file)
                embeddings_model = OpenAIEmbeddings(openai_api_key=openai.api_key)
                vectordb = embed_and_store(pages, embeddings_model)
                st.session_state.retriever = vectordb.as_retriever()
                mark_as_processed(file_name)
                st.sidebar.success("PDF Processed and Stored!")
                st.session_state.pdf_processed = True
        else:
            if 'retriever' not in st.session_state:
                with st.sidebar.spinner("Loading existing data..."):
                    index_name = PINECONE_INDEX_NAME
                    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
                    docsearch = Pinecone.from_existing_index(index_name, embeddings)
                    st.session_state.retriever = docsearch.as_retriever()
                    st.info("PDF already processed. Using existing data.")
                    st.session_state.pdf_processed = True
    return None


def extract_text_from_pdf(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        pages = [page.extract_text() for page in pdf.pages]
    return pages

def embed_and_store(pages, embeddings_model):
    # Embedding the documents and storing them in Pinecone
    docsearch = Pinecone.from_texts(pages, embeddings_model, index_name=PINECONE_INDEX_NAME)
    return docsearch

def save_questions_to_file(questions, filename="generated_questions.txt", num_questions=20):
    # Ensure we don't exceed the number of available questions
    num_questions = min(num_questions, len(questions))

    # Write the questions to the file
    with open(filename, 'w') as file:
        for question in questions[:num_questions]:
            file.write(question + '\n')

    # Optionally, you can also display the questions in the Streamlit app
    st.write("Generated Questions:")
    for question in questions[:num_questions]:
        st.write(question)


def has_been_processed(file_name):
    """Check if the PDF has already been processed."""
    processed_files = set()
    if os.path.exists("processed_files.txt"):
        with open("processed_files.txt", "r") as file:
            processed_files = set(file.read().splitlines())
    return file_name in processed_files

def mark_as_processed(file_name):
    """Mark the PDF as processed."""
    with open("processed_files.txt", "a") as file:
        file.write(file_name + "\n")

def handle_enter():
    if 'retriever' in st.session_state:
        user_input = st.session_state.user_input
        if user_input:
            st.session_state.chat_history.append(("You", user_input))
            with st.spinner("Please wait..."):  # Show a loading spinner
                try:
                    qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=st.session_state.retriever)
                    bot_response = qa.run(user_input)
                    st.session_state.chat_history.append(("Bot", bot_response))
                except Exception as e:
                    st.session_state.chat_history.append(("Bot", f"Error - {e}"))
            st.session_state.user_input = ""  # Clear the input box after processing

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

if __name__ == "__main__":
    main()
