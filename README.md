# 📄 DocuMind AI - Intelligent Document Analysis

## 🚀 Overview

**DocuMind AI** is a powerful document processing and analysis tool that leverages **Google's Gemini AI** and **LangChain** to provide intelligent insights from your documents. It supports PDFs and images, offering features like **text extraction, PII detection, redaction, and natural language Q&A** about document content.

🔗 **Live Demo**: [Documind-AI](https://docuwiz-ai-prerna.streamlit.app/)

---

## ✨ Features

- 📂 **Multi-Format Support** – Process both PDFs and images (PNG, JPG, JPEG)
- 🔍 **Advanced Text Extraction** – Extract text from scanned documents using OCR (Tesseract)
- 🔒 **PII Detection & Redaction** – Automatically find and hide sensitive information (emails, phones, credit cards)
- 💬 **Document Q&A** – Ask natural language questions about your documents
- 📊 **Export Options** – Download processed content in TXT, PDF, or DOCX formats
- 🧠 **AI-Powered Analysis** – Uses Google Gemini for intelligent document understanding
- 🛡️ **Privacy-Focused** – All processing happens locally (except Gemini API calls)

---

## 🛠 Tech Stack

| **Core Technologies**      | **Supporting Libraries**         |
|-----------------------------|----------------------------------|
| Python (3.8+)               | PyPDF2 (PDF Extraction)         |
| Streamlit (Frontend)        | PyMuPDF (PDF Rendering)         |
| Google Gemini (AI)          | Tesseract OCR (Image Processing)|
| LangChain (Document Processing)| python-docx (Word Export)     |
| FAISS (Vector Storage)      |                                  |


---

## 📂 Project Structure
```bash
DocuMind-AI/
├── app.py                # Main application
├── requirements.txt      # Dependencies
├── .env                  # Environment variables
├── faiss_index/          # Vector store directory
└── outputs/              # Processed outputs

```

## 🏁 Getting Started

### Prerequisites

- Python 3.8+
- Tesseract OCR installed (for image processing)
- Google Gemini API key

### 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/prerna-patil26/DocuMind-AI.git
cd DocuMind-AI

```
2. Install dependencies:
```bash
pip install -r requirements.txt

```

3. Set up environment
```bash
echo "GEMINI_API_KEY=your_api_key" > .env
```

4. Run the app:
```bash
streamlit run app.py
```

## 👩‍💻 Author

- **Prerna Patil**
  - 🎓 MCA Student
  - 🧠 AI & ML Enthusiast
  - 📧 Email: `prernapatil2608@gmail.com`
  - 🔗 LinkedIn: [Prerna Patil](https://www.linkedin.com/in/prerna-patil26) <!-- Replace # with your LinkedIn URL -->

---

## ⭐ Support the Project

- 🌟 Give it a ⭐ on [GitHub](https://github.com/prerna-patil26/DocuMind-AI)
- 📢 Share with your network
- 🤝 Contribute to development
- 🐞 Report issues or suggestions
