import streamlit as st
from PyPDF2 import PdfReader
import os
import pytesseract
from PIL import Image
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
import io
from fpdf import FPDF
import docx
import re
import base64
import fitz  # PyMuPDF
import shutil
import glob
from streamlit.components.v1 import html

# Configure Tesseract path
current_dir = os.path.dirname(os.path.abspath(__file__))
pytesseract.pytesseract.tesseract_cmd = os.path.join(current_dir, "tesseract", "tesseract.exe")


# Load environment variables
load_dotenv()

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'processed_text' not in st.session_state:
    st.session_state.processed_text = ""
if 'show_guide' not in st.session_state:
    st.session_state.show_guide = False
if 'download_format' not in st.session_state:
    st.session_state.download_format = "TXT"
if 'original_files' not in st.session_state:
    st.session_state.original_files = None

# Custom CSS for styling
st.markdown("""
<style>

/* 1. Light blue shadow on all boxes */
.card, .chat-container, .guide-container, .download-container, .upload-area {
    box-shadow: 0 4px 12px rgba(0, 123, 255, 0.1);
    border-radius: 12px;
}

/* 🔧 Compact Sidebar Buttons and Spacing */
section[data-testid="stSidebar"] {
    padding: 1rem 0.75rem !important;
}

/* Reduce top space and spacing between all sidebar elements */
section[data-testid="stSidebar"] > div {
    padding-top: 0 !important;
    gap: 0.3rem !important;
    margin-bottom: 0.2rem !important;
}

/* 🔹 Smaller buttons + reduced spacing */
section[data-testid="stSidebar"] .stButton > button {
    font-size: 0.75rem !important;
    padding: 4px 10px !important;
    border-radius: 16px !important;
    margin: 4px auto !important;
    width: 100% !important;
}

/* Smaller selectbox and download dropdown */
section[data-testid="stSidebar"] .stSelectbox,
section[data-testid="stSidebar"] .stDownloadButton,
section[data-testid="stSidebar"] .stTextInput {
    font-size: 0.75rem !important;
    padding: 4px 10px !important;
}

/*Subheader and headings smaller */
.sidebar-header {
    font-size: 1.1rem !important;
    margin-bottom: 0.2rem !important;
}
.sidebar-subheader {
    font-size: 0.75rem !important;
    margin-bottom: 0.6rem !important;
}

/* 🧼 Remove excess space above file uploader */
div[data-testid="stFileUploader"] {
    margin-top: 0.25rem !important;
}


button, .stButton>button, .back-btn {
    background: linear-gradient(90deg, #7f5af0, #3c8ce7);
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 0.5rem 1rem;
    transition: all 0.3s ease;
}

button, .stButton > button {
  border-radius: 25px !important;
  padding: 6px 16px !important;
  font-size: 0.85rem !important;
  background: linear-gradient(to right, #7b1fa2, #2196f3);
  color: white;
  border: none;
  margin: 8px auto;
  display: block;
  width: fit-content;
}

section[data-testid="stSidebar"] .stButton > button {
  margin-left: auto;
  margin-right: auto;
}
            

button[kind="secondary"] {
  max-width: 200px;
  margin: auto;
  font-size: 0.85rem;
  padding: 6px 14px;
  border-radius: 20px !important;
}

section[data-testid="stSidebar"] .stButton {
  margin-bottom: 0.4rem !important;
}


section[data-testid="stSidebar"] {
    min-width: 320px !important;
    max-width: 360px !important;
}


button:hover, .stButton>button:hover {
    opacity: 0.9;
}

/* Sidebar header styling */
    .sidebar-header {
        background: linear-gradient(to right, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 8px rgba(59, 130, 246, 0.4);
 }

.sidebar-subheader {
        color: linear-gradient(to right, #3b82f6, #8b5cf6);
        font-size: 0.85rem;
        margin-bottom: 1rem;
        font-weight: 500;
}
    
    /* Dark mode adjustment */
    @media (prefers-color-scheme: dark) {
        .sidebar-subheader {
            color: #94a3b8;
        }
    }


.upload-container .stButton>button {
    background: linear-gradient(90deg, #7f5af0, #3c8ce7);
    color: white;
    border-radius: 10px;
    padding: 0.4rem 1rem;
    font-weight: 600;
}

section[data-testid="stSidebar"] > div:first-child {
    min-width: 300px;
    max-width: 340px;
}


section[data-testid="stSidebar"]::after {
    content: "";
    position: absolute;
    top: 0;
    right: 0;
    width: 3px;
    height: 100%;
    background: linear-gradient(to bottom, #7b1fa2, #2196f3); /* Gradient purple to blue */
    box-shadow: 0 0 8px rgba(123, 31, 162, 0.7), 0 0 12px rgba(33, 150, 243, 0.7);
    z-index: 5;
}
section[data-testid="stSidebar"] {
    position: relative;
}
            
section[data-testid="stSidebar"] > div {
  gap: 0.5rem !important;
  margin-bottom: 0.25rem !important;
}
 
section[data-testid="stSidebar"] {
    background-color: #000000 !important;  /* Pure black */
    color: white;
    padding: 1.5rem 1rem;
    position: relative;
    box-shadow: inset -3px 0 6px rgba(255, 255, 255, 0.05);
}


section[data-testid="stSidebar"] * {
    color: white !important;
}


.download-label::before {
    content: "\25BC "; /* Down arrow icon */
    margin-right: 6px;
}
.download-label {
    font-weight: bold;
    margin-bottom: 6px;
    color: #444;
}

 /* Reduce gap between User Guide and Reset buttons */
    div.stButton:has(button[kind="secondary"]) {
        margin-top: 0.05rem !important;
    }
.header {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(to right, #7f5af0, #3c8ce7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 1px 1px 4px rgba(0, 0, 0, 0.2);
    margin-bottom: 1rem;
    text-align: center;
}


.upload-section {
  background: linear-gradient(to right, #7b1fa2, #2196f3);
  color: white;
  padding: 12px;
  border-radius: 20px;
  margin-top: 1rem;
  text-align: center;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 1rem;
  flex-wrap: wrap;
}
.upload-section span {
  font-weight: 600;
  font-size: 1rem;
}
.upload-section .browse-button button {
  border-radius: 25px !important;
  padding: 6px 16px !important;
  font-size: 0.85rem !important;
  background: linear-gradient(to right, #7b1fa2, #2196f3);
  color: white;
  border: none;
}
  

div[data-testid="stFileUploader"] > div:first-child {
    background: linear-gradient(to right, #7b1fa2, #2196f3) !important;
    border-radius: 20px !important;
    padding: 1rem 1.2rem !important;
    color: white !important;
    font-weight: 600;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
    margin-top: 0.5rem;
    text-align: center;
    margin-top: 0 !important;
    padding-top: 0 !important;
}

div[data-testid="stFileUploader"] label::before {
    content: "📂";
    margin-right: 8px;
    font-size: 1.2rem;
}


div[data-testid="stFileUploader"] button {
    background: linear-gradient(to right, #7b1fa2, #2196f3) !important;
    color: white !important;
    border-radius: 25px !important;
    padding: 6px 16px !important;
    font-size: 0.85rem !important;
    border: none !important;
    margin-top: 1rem;
    transition: 0.3s ease;
}
div[data-testid="stFileUploader"] button:hover {
    opacity: 0.9;
}

  .chat-container {
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
    border: 2px solid transparent;
    background-clip: padding-box;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}
.user-message {
    border: 2px solid #64b5f6; /* Light blue */
    background-color: rgba(100, 181, 246, 0.1);
    padding: 1rem;
    border-radius: 10px;
}
.bot-message {
    border: 2px solid #9575cd; /* Light purple */
    background-color: rgba(149, 117, 205, 0.1);
    padding: 1rem;
    border-radius: 10px;
}

.upload-container .stButton:last-child {
    margin-left: auto;
}

@media (prefers-color-scheme: dark) {
    .chat-container, .guide-container, .upload-area {
        background-color: rgba(30, 30, 30, 0.95);
    }
}
            
</style>
""", unsafe_allow_html=True)

def extract_text_from_pdf(pdf_file):
    try:
        doc = fitz.open(stream=pdf_file.getvalue(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"PDF processing error: {str(e)}")
        return ""

def extract_text_from_image(image_file):
    try:
        img = Image.open(image_file)
        text = pytesseract.image_to_string(img)
        return text if text.strip() else "No text could be extracted from the image"
    except Exception as e:
        return f"Image processing error: {str(e)}"

def extract_text_from_docx(docx_file):
    try:
        doc = docx.Document(io.BytesIO(docx_file.getvalue()))
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        return f"DOCX processing error: {str(e)}"

def extract_text_from_txt(txt_file):
    try:
        return txt_file.getvalue().decode("utf-8")
    except Exception as e:
        return f"Text file processing error: {str(e)}"

def get_pdf_text(uploaded_files):
    text = ""
    for file in uploaded_files:
        if file.name.lower().endswith(('.png', '.jpg', '.jpeg')):
            st.info(f"Processing image: {file.name}")
            file_text = extract_text_from_image(file)
            text += f"Image Content:\n{file_text}\n\n"
        elif file.name.lower().endswith('.pdf'):
            st.info(f"Processing PDF: {file.name}")
            file_text = extract_text_from_pdf(file)
            text += f"PDF Content:\n{file_text}\n\n"
        elif file.name.lower().endswith('.docx'):
            st.info(f"Processing DOCX: {file.name}")
            file_text = extract_text_from_docx(file)
            text += f"DOCX Content:\n{file_text}\n\n"
        elif file.name.lower().endswith('.txt'):
            st.info(f"Processing TXT: {file.name}")
            file_text = extract_text_from_txt(file)
            text += f"TXT Content:\n{file_text}\n\n"
        else:
            st.warning(f"Unsupported file type: {file.name}")
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    st.session_state.vector_store = vector_store
    st.session_state.processed_text = "\n".join(text_chunks)
    return vector_store

def get_conversational_chain():
    prompt_template = """
    You are an expert document assistant named DocuMind. Follow these rules strictly:
    
    1. Provide detailed, accurate answers from the document context
    2. Only include page numbers when explicitly asked by the user
    3. For "what page" or "which page" questions, always provide the page number
    4. Structure answers clearly with bullet points when appropriate
    5. If information isn't found, say "This information was not found in the document"
    
    Context:\n{context}\n
    Question: {question}\n
    
    Response Examples:
    
    User asks: "What is the main conclusion?"
    Response: "The main conclusion is that... (additional details)"
    
    User asks: "What page is the methodology on?"
    Response: "The methodology is described on page 5."
    
    User asks: "Tell me about the results"
    Response: "The results show:
    • Key finding 1 (details)
    • Key finding 2 (details)"
    """
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def detect_pii(text):
    try:
        results = {
            "emails": re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text),
            "phones": re.findall(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', text),
            "credit_cards": re.findall(r'\b(?:\d[ -]*?){13,16}\b', text),
            "names": re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', text)
        }
        return results
    except Exception as e:
        st.error(f"PII detection error: {str(e)}")
        return None

def redact_pdf_content(pdf_file):
    try:
        doc = fitz.open(stream=pdf_file.getvalue(), filetype="pdf")
        for page in doc:
            text = page.get_text()
            pii_results = detect_pii(text)
            if pii_results:
                for email in pii_results.get("emails", []):
                    areas = page.search_for(email)
                    [page.add_redact_annot(area, fill=(0, 0, 0)) for area in areas]
                for phone in pii_results.get("phones", []):
                    areas = page.search_for(phone)
                    [page.add_redact_annot(area, fill=(0, 0, 0)) for area in areas]
                page.apply_redactions()
        return doc
    except Exception as e:
        st.error(f"PDF redaction error: {str(e)}")
        return None

def create_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    # Handle UTF-8 characters properly
    text = text.encode('latin1', 'replace').decode('latin1')
    pdf.multi_cell(0, 10, text)
    return pdf.output(dest='S').encode('latin1')

def export_content(content, format='txt', filename="export"):
    try:
        if format == 'txt':
            return content.encode('utf-8'), 'text/plain', f"{filename}.txt"
        elif format == 'pdf':
            return create_pdf(content), 'application/pdf', f"{filename}.pdf"
        elif format == 'docx':
            doc = docx.Document()
            doc.add_paragraph(content)
            bio = io.BytesIO()
            doc.save(bio)
            return bio.getvalue(), 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', f"{filename}.docx"
    except Exception as e:
        st.error(f"Export error: {str(e)}")
        return None, None, None

def reset_all_data():
    # Clear FAISS index
    try:
        shutil.rmtree("faiss_index", ignore_errors=True)
        st.success("✅ FAISS index cleared.")
    except Exception as e:
        st.error(f"Error clearing FAISS index: {str(e)}")
    
    # Clear session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    st.success("Chat history and document data cleared. Please upload new documents to continue.")
    st.rerun()

def show_guide():
    st.markdown("""
    ## 📚 DocuMind AI Guide
    
    ### ✨ Core Features
    
    **Document Analysis**
    - Extract key information from PDFs, DOCX, TXT, and images
    - Get precise answers to your document questions
    - Automatically detect page references when requested
    
    **Smart Summarization**
    - Generate summaries of any length (50-1000 words)
    - Focus on specific sections when needed
    - Maintain all critical information
    
    **PII Detection & Redaction**
    - Identify sensitive information:
      - Email addresses
      - Phone numbers
      - Credit card information
    - One-click redaction and export
    
    ### 🚀 How To Use
    
    1. **Upload Documents**
       - PDF, DOCX, TXT, JPG, PNG files supported
       - Multiple files can be processed together
    
    2. **Ask Questions**
       - "What's the main conclusion?"
       - "Explain the methodology on page 5"
       - "List all key findings"
    
    3. **Generate Summaries**
       - "200-word summary of the document"
       - "Brief overview of the introduction"
       - "50-word summary of the results section"
    
    4. **Handle Sensitive Data**
       - "Find all email addresses"
       - "Redact phone numbers and export"
       - "Show detected PII"
    """)
    
    if st.button("⬅️ Back to Chat", key="back_from_guide"):
        st.session_state.show_guide = False
        st.rerun()

def main():
    st.set_page_config("✨ DocuMind AI", page_icon="📄", layout="wide")
    st.markdown('<div class="header">✨DocuMind Ai — Chat with Documents</div>', unsafe_allow_html=True)

    if not st.session_state.get('processed_text') and not st.session_state.get('show_guide'):
        st.markdown("<h1 style='text-align: center; margin-top: 2rem;'>Welcome to DocuMind AI</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Upload and process your documents to begin</p>", unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-header">Documind AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-subheader">Upload documents and images</div>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=['pdf', 'docx', 'txt', 'png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        
        if st.button("Process Documents", use_container_width=True):
            with st.spinner("Processing..."):
                if uploaded_files:
                    st.session_state.original_files = uploaded_files
                    raw_text = get_pdf_text(uploaded_files)
                    if raw_text.strip():
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Processing complete!")
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": "Hello! I've processed your documents. How can I help you today?"
                        })
                        st.rerun()
                    else:
                        st.error("No text extracted - check file formats")
                else:
                    st.warning("Please upload files first")
        
        st.divider()
        
        # Download options
        if st.session_state.processed_text:
            st.subheader("Export Options")
            col1, col2 = st.columns([3, 1])
            with col1:
                st.session_state.download_format = st.selectbox(
                    "Format",
                    ["TXT", "PDF", "DOCX"],
                    label_visibility="collapsed"
                )
            with col2:
                export_filename = "document_analysis"
                content = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.chat_history])
                
                file_data, mime_type, file_name = export_content(
                    content,
                    st.session_state.download_format.lower(),
                    export_filename
                )
                
                if file_data:
                    st.download_button(
                        label="⬇️",
                        data=file_data,
                        file_name=file_name,
                        mime=mime_type,
                        use_container_width=True
                    )
        
        st.divider()
        
        if st.button("📘 User Guide", use_container_width=True):
            st.session_state.show_guide = True
            st.rerun()
        
        st.divider()
        
        if st.button("🔄 Reset All Data", 
                   use_container_width=True, 
                   type="secondary",
                   help="Clear all chat history and document data"):
            reset_all_data()
    
    # Main content
    if st.session_state.show_guide:
        show_guide()
    else:
        # Chat interface
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-container user-message">
                    <div>
                        <strong>You:</strong> {message["content"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-container bot-message">
                    <div>
                        <strong>DocuMind:</strong> {message["content"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # User input
        user_question = st.chat_input("Ask about your documents...")
        if user_question and st.session_state.vector_store:
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            with st.spinner("Analyzing..."):
                try:
                    # Handle PII requests
                    if "pii" in user_question.lower() or "hide" in user_question.lower() or "redact" in user_question.lower():
                        if st.session_state.original_files:
                            # Process only the first PDF for redaction (can be extended for multiple files)
                            pdf_file = next((f for f in st.session_state.original_files if f.name.lower().endswith('.pdf')), None)
                            if pdf_file:
                                redacted_doc = redact_pdf_content(pdf_file)
                                if redacted_doc:
                                    pdf_bytes = redacted_doc.write()
                                    b64 = base64.b64encode(pdf_bytes).decode()
                                    pdf_download = f'<a href="data:application/pdf;base64,{b64}" download="redacted_document.pdf">⬇️ Download Redacted PDF</a>'
                                    answer = f"I've redacted PII from the original document. {pdf_download}"
                                else:
                                    answer = "Could not redact the document. Please try again."
                            else:
                                answer = "No PDF file found to redact."
                        else:
                            answer = "No original files available for redaction. Please upload documents again."
                    
                    # Handle general questions
                    else:
                        chain = get_conversational_chain()
                        docs = st.session_state.vector_store.similarity_search(user_question)
                        response = chain(
                            {"input_documents": docs, "question": user_question},
                            return_only_outputs=True
                        )
                        answer = response['output_text']
                    
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    st.rerun()
                except Exception as e:
                    st.error(f"Analysis error: {str(e)}")
    
    # JavaScript for functionality
    st.markdown("""
    <script>
    function backToChat() {
        window.parent.postMessage({type: 'streamlit:setComponentValue', value: 'back_to_chat'}, '*');
    }
    </script>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()