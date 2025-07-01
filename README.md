---
title: PDF Chat App
emoji: "📄"
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.45.1
app_file: app.py
pinned: false
---

# PDF Chat App

A modern, privacy-focused web app to chat with your PDF documents using AI. Upload any PDF, ask questions, and get instant answers powered by state-of-the-art language models. Built with Streamlit, LangChain, and Hugging Face Transformers. Easily deployable on [Hugging Face Spaces](https://huggingface.co/spaces/saketh-005/pdf-chat-app) or run locally.

---

## ✨ Features
- 📄 Upload and process PDF documents
- 💬 Chat with your documents using natural language
- 🔒 100% local processing (no data leaves your machine)
- 🤗 Uses Hugging Face models for embeddings and question answering
- 🚀 One-click deployment on Hugging Face Spaces
- 🖥️ Simple, beautiful Streamlit interface

---

## 🚀 Getting Started

### On Hugging Face Spaces
Just visit: [https://huggingface.co/spaces/saketh-005/pdf-chat-app](https://huggingface.co/spaces/saketh-005/pdf-chat-app)

### Run Locally
1. **Clone the repository:**
   ```bash
   git clone https://github.com/saketh-005/pdf-chat-app.git
   cd pdf-chat-app
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Start the app:**
   ```bash
   streamlit run app.py
   ```
4. **Open your browser:**
   Go to [http://localhost:8501](http://localhost:8501)

---

## 🖥️ Usage
1. Click "Browse files" to upload a PDF document
2. Wait for the document to be processed
3. Type your question in the chat input and press Enter
4. The app will analyze the document and provide an answer

---

## 🏗️ Project Structure
```
.
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .gitignore          # Git ignore file
├── LICENSE             # License file
└── README.md           # This file
```

---

## 🤖 Technologies Used
- [Streamlit](https://streamlit.io/) - Web application framework
- [LangChain](https://python.langchain.com/) - Framework for LLM applications
- [Hugging Face Transformers](https://huggingface.co/transformers/) - NLP models
- [Chroma DB](https://www.trychroma.com/) - Vector database for document storage

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Credits
- [Hugging Face](https://huggingface.co/) for their amazing open-source models
- [LangChain](https://python.langchain.com/) for simplifying LLM application development
- [Streamlit](https://streamlit.io/) for the intuitive web interface

---

**Author:** Saketh Jangala
