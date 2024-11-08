import streamlit as st
from azure.storage.blob import BlobServiceClient
import openai
import pandas as pd
import PyPDF2
import re
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from docx import Document
import mammoth 

# Set up Azure Blob Storage
AZURE_BLOB_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
BLOB_CONTAINER_NAME = os.getenv("BLOB_CONTAINER_NAME")
blob_service_client = BlobServiceClient.from_connection_string(AZURE_BLOB_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)

# Azure OpenAI setup using OpenAI Python library
openai.api_type = "azure"
openai.api_base = os.getenv("api_base")
openai.api_version = "2023-03-15-preview"
openai.api_key = os.getenv("api_key")

MODEL_DEPLOYMENT_NAME = "gpt-4" 
EMBEDDING_DEPLOYMENT_NAME = "text-embedding-ada-002" 

# Streamlit UI
st.title("💬 GenAi Chatbot")
st.write("Chat with the bot, upload a file, and leverage embedded content for responses.")

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You can ask questions or upload files for analysis."}]
if "file_embeddings" not in st.session_state:
    st.session_state.file_embeddings = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "submitted" not in st.session_state:
    st.session_state.submitted = False

def clean_text(text):
    """Clean and normalize the text."""
    return re.sub(r"[\r\n\t\f]+", " ", text).strip()

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF files."""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return clean_text(text)

def extract_text_from_docx(docx_file):
    """Extract text from DOCX files."""
    doc = Document(docx_file)
    text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
    return clean_text(text)

def extract_text_from_doc(doc_file):
    """Extract text from DOC files using Mammoth."""
    try:
        with doc_file as file:
            result = mammoth.extract_raw_text(file)
        return clean_text(result.value)
    except Exception as e:
        return f"Error extracting text from .doc file: {e}"

def embed_text(text):
    """Embed text using Azure OpenAI's text-embedding-ada-002."""
    try:
        response = openai.Embedding.create(
            input=text,
            engine=EMBEDDING_DEPLOYMENT_NAME
        )
        return response['data'][0]['embedding']
    except Exception as e:
        st.error(f"Error embedding text: {e}")
        return None

def split_into_chunks(text, chunk_size=500):
    """Split text into manageable chunks."""
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        if len(" ".join(current_chunk + [word])) <= chunk_size:
            current_chunk.append(word)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def search_embedded_content(query):
    """Search for the most relevant embedded content."""
    if not st.session_state.file_embeddings:
        return None, 0

    query_embedding = embed_text(query)
    if query_embedding is None:
        return None, 0

    similarities = [cosine_similarity([query_embedding], [item["embedding"]])[0][0]
                    for item in st.session_state.file_embeddings]
    best_match_idx = np.argmax(similarities)
    best_score = similarities[best_match_idx]

    if best_score > 0.75:  # Threshold for relevance
        return st.session_state.file_embeddings[best_match_idx]["content"], best_score
    return None, best_score

def enhance_answer_with_gpt(query, context):
    """Enhance the response using GPT-4 by combining query and relevant context."""
    try:
        response = openai.ChatCompletion.create(
            engine=MODEL_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Context: {context}\n\nQuery: {query}"}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        st.error(f"Error with GPT-4 response: {e}")
        return "An error occurred while generating the response."

def submit_message(user_message):
    """Handle user query and provide a response."""
    st.session_state.messages.append({"role": "user", "content": user_message})

    try:
        # Search for relevant content
        result, score = search_embedded_content(user_message)
        if result:
            # Enhance the response with GPT-4
            enhanced_response = enhance_answer_with_gpt(user_message, result)
            st.session_state.messages.append({"role": "assistant", "content": enhanced_response})
        else:
            # Fallback to GPT-4 for general responses
            general_response = openai.ChatCompletion.create(
                engine=MODEL_DEPLOYMENT_NAME,
                messages=st.session_state.messages,
                max_tokens=500,
                temperature=0.7
            )['choices'][0]['message']['content']
            st.session_state.messages.append({"role": "assistant", "content": general_response})

    except Exception as e:
        st.session_state.messages.append({"role": "error", "content": f"Error: {e}"})
        st.error(f"Chat processing error: {e}")

chat_style = """
    <style>
    .chat-history {
        margin-bottom: 10px;
    }
    .chat-message {
        background-color: white;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0px 0px 5px rgba(0,0,0,0.1);
        max-height: 400px;
        overflow-y: auto;
    }
    .user-message {
        font-weight: bold;
        color: #1a73e8;  /* Blue color for user */
    }
    .bot-message {
        font-weight: bold;
        color: #34a853;  /* Green color for bot */
    }
    </style>
"""

st.markdown(chat_style, unsafe_allow_html=True)
st.markdown("### Chat History")
st.markdown('<div class="chat-history">', unsafe_allow_html=True)

# Display chat history with styled messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-message user-message">You: {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-message bot-message">GenAi: {msg["content"]}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Form for unified input and button
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Enter your message:", placeholder="Type your message here...", key="chat_input")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    submit_message(user_input)
    st.rerun()

# File upload and embedding
uploaded_file = st.file_uploader("Upload a file (txt, csv, xlsx, pdf, docx, doc)", type=["txt", "csv", "xlsx", "pdf", "docx", "doc"])
if uploaded_file:
    try:
        if uploaded_file.type == "text/plain":
            file_content = uploaded_file.read().decode("utf-8")
        elif uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
            file_content = df.to_csv(index=False)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            df = pd.read_excel(uploaded_file)
            file_content = df.to_csv(index=False)
        elif uploaded_file.type == "application/pdf":
            file_content = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            file_content = extract_text_from_docx(uploaded_file)
        elif uploaded_file.type == "application/msword":
            file_content = extract_text_from_doc(uploaded_file)

        # Split into chunks and embed
        chunks = split_into_chunks(file_content)
        st.session_state.file_embeddings = [
            {"content": chunk, "embedding": embed_text(chunk)} for chunk in chunks if chunk.strip()
        ]

        # Save file to Azure Blob Storage
        blob_client = container_client.get_blob_client(uploaded_file.name)
        blob_client.upload_blob(uploaded_file, overwrite=True)
        st.success(f"File '{uploaded_file.name}' uploaded and embedded successfully.")
        st.session_state.uploaded_files.append(uploaded_file.name)

    except Exception as e:
        st.error(f"Error processing file: {e}")

# Display and allow selection of previously uploaded files
st.markdown("### Previously Uploaded Files")
blobs = container_client.list_blobs()
for blob in blobs:
    if st.button(f"Load {blob.name}"):
        blob_client = container_client.get_blob_client(blob.name)
        file_content = blob_client.download_blob().readall().decode('utf-8')
        chunks = split_into_chunks(file_content)
        st.session_state.file_embeddings = [
            {"content": chunk, "embedding": embed_text(chunk)} for chunk in chunks if chunk.strip()
        ]
        st.success(f"Loaded and embedded content from '{blob.name}'.")