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
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

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
    .header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #FFEB3B; /* Light yellow */
        margin-bottom: 1rem;
        text-align: center;
    }
    .chat-container {
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        background-color: rgba(255, 255, 255, 0.9);
        border: 1px solid #FFD166; /* Light yellow-orange */
    }
    .user-message {
        color: #000000;
    }
    .bot-message {
        color: #000000;
    }
    .download-container {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .guide-container {
        padding: 2rem;
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        position: relative;
    }
    .back-btn {
        background-color: #4f46e5;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 4px;
        cursor: pointer;
        margin-top: 1rem;
    }
    .reset-note {
        font-size: 0.8rem;
        color: #666;
        margin-top: 0.5rem;
    }
    @media (prefers-color-scheme: dark) {
        .chat-container, .guide-container {
            background-color: rgba(30, 30, 30, 0.9);
        }
        .user-message, .bot-message {
            color: #FFFFFF;
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

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        if pdf.name.lower().endswith(('.png', '.jpg', '.jpeg')):
            st.info(f"Processing image: {pdf.name}")
            file_text = extract_text_from_image(pdf)
            text += f"Image Content:\n{file_text}\n\n"
        elif pdf.name.lower().endswith('.pdf'):
            st.info(f"Processing PDF: {pdf.name}")
            file_text = extract_text_from_pdf(pdf)
            text += f"PDF Content:\n{file_text}\n\n"
        else:
            st.warning(f"Unsupported file type: {pdf.name}")
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
    - Extract key information from PDFs and images
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
       - PDF, JPG, PNG files supported
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

    
    # Header with light yellow color
    st.markdown('<div class="header">✨DocuMind Ai — Chat with PDFs & Images </div>', unsafe_allow_html=True)

     # 1. Simple Centered Welcome Message
    if not st.session_state.get('processed_text') and not st.session_state.get('show_guide'):
        st.markdown("<h1 style='text-align: center; margin-top: 2rem;'>Welcome to DocuMind AI</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Upload and process your documents to begin</p>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.subheader("📂 Document Processing")
        uploaded_files = st.file_uploader(
            "Upload PDFs/Images",
            type=['pdf', 'png', 'jpg', 'jpeg'],
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