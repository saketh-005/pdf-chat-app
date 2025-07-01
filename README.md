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

# PDF Chat Application

A PDF chat application that allows you to upload PDFs and ask questions about their content using natural language processing. Built with Streamlit, LangChain, and Hugging Face Transformers, this app runs entirely in your browser on Hugging Face Spaces.

## ✨ Features

- 📄 Upload and process PDF documents
- 💬 Chat with your documents using natural language
- 🔒 Local processing - no data leaves your machine
- 🤗 Uses Hugging Face models for embeddings and question answering
- 🚀 Built with Streamlit for a clean web interface

## 🛠 Prerequisites

- A Hugging Face account (for Spaces deployment)
- Git (for cloning the repository)
- At least 4GB of free RAM (for running the models)

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/saketh-005/pdf-chat-app.git
   cd pdf-chat-app
   ```

2. Install dependencies and run locally (optional):
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```

3. Or deploy directly to Hugging Face Spaces by pushing this folder to your Space.

## 🖥️ Usage

1. Click "Browse files" to upload a PDF document
2. Wait for the document to be processed (you'll see a success message)
3. Type your question in the chat input and press Enter
4. The app will analyze the document and provide an answer

## 🏗️ Project Structure

```
.
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .gitignore          # Git ignore file
└── README.md           # This file
```

## 🤖 Technologies Used

- [Streamlit](https://streamlit.io/) - Web application framework
- [LangChain](https://python.langchain.com/) - Framework for LLM applications
- [Hugging Face Transformers](https://huggingface.co/transformers/) - NLP models
- [Chroma DB](https://www.trychroma.com/) - Vector database for document storage

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for their amazing open-source models
- [LangChain](https://python.langchain.com/) for simplifying LLM application development
- [Streamlit](https://streamlit.io/) for the intuitive web interface
