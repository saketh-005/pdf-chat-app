import os
import sys
import torch
import streamlit as st
from PyPDF2 import PdfReader
from typing import List, Dict, Any, Optional

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

# Transformers imports
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
    set_seed
)

# Set random seed for reproducibility
set_seed(42)

# Disable HuggingFace warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    text = ""
    try:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        if not text.strip():
            st.error("Could not extract any text from the PDF. The PDF might be scanned or protected.")
            return None
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {str(e)}")
        return None

def generate_response(uploaded_file, query_text):
    """
    Handles the main logic using local Hugging Face models.
    No API key required as everything runs locally.
    """
    if uploaded_file is None:
        return "Error: No file uploaded."

    # 1. Extract text from PDF
    st.info("Reading your PDF document...")
    raw_text = extract_text_from_pdf(uploaded_file)
    if raw_text is None:
        return "Error: Could not extract text from the PDF."

    # 2. Split text into manageable chunks
    st.info("Splitting text into chunks...")
    # Split the text into chunks with attention to model's max sequence length (512 tokens)
    # Using a conservative chunk size to account for tokenization differences
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,  # Reduced from 1000 to stay well under 512 tokens
        chunk_overlap=100,  # Slightly reduced overlap
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", ". ", " ", ""],  # Added explicit separators
    )
    texts = text_splitter.split_text(raw_text)

    # 3. Create embeddings and vector store
    st.info("Creating document embeddings...")
    
    # Use GPU if available, otherwise CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # Try to use a more powerful embeddings model first
        embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-mpnet-base-v2',
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        # Test the embeddings model
        test_emb = embeddings.embed_query("test")
        if not test_emb or len(test_emb) == 0:
            raise Exception("Embeddings model returned empty result")
            
    except Exception as e:
        st.warning(f"Falling back to smaller embeddings model due to: {str(e)}")
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name='sentence-transformers/all-MiniLM-L6-v2',
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            st.error(f"Failed to load embeddings model: {str(e)}")
            return "Error: Could not load embeddings model."
    
    try:
        # Create ChromaDB vector store with metadata
        try:
            document_search = Chroma.from_texts(
                texts=texts,
                embedding=embeddings,
                metadatas=[{"source": f"chunk-{i}", "page": i+1} for i in range(len(texts))],
                collection_metadata={"hnsw:space": "cosine"}
            )
            # Test the vector store
            _ = document_search.similarity_search("test", k=1)
        except Exception as e:
            st.error(f"Error creating vector store: {str(e)}")
            st.stop()
        
        # Force a small operation to verify the vector store works
        _ = document_search.similarity_search("test", k=1)
        
    except Exception as e:
        st.error(f"Failed to create vector store: {str(e)}")
        st.exception(e)  # Show full traceback for debugging
        return "Error: Could not process document content."
    
    # 4. Load the question-answering model
    st.info("Loading question-answering model...")
    
    # Model selection with fallback
    model_name = "google/flan-t5-large"
    fallback_model = "google/flan-t5-base"
    
    try:
        # Try to use the base model first
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )
    except Exception as e:
        st.warning(f"Falling back to smaller model due to: {str(e)}")
        try:
            model_name = fallback_model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=torch.float32,  # Use float32 for stability on CPU
                low_cpu_mem_usage=True
            )
        except Exception as e:
            st.error(f"Failed to load language model: {str(e)}")
            return "Error: Could not load question-answering model."
    
    try:
        # Create text generation pipeline
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=1024,
            temperature=0.2,
            do_sample=True,
            top_p=0.92,
            top_k=50,
            num_beams=4,
            device=0 if torch.cuda.is_available() else -1,
        )
        
        llm = HuggingFacePipeline(
            pipeline=pipe,
            model_kwargs={
                "temperature": 0.2,
                "max_length": 1024,
                "repetition_penalty": 1.2,
                "no_repeat_ngram_size": 3
            }
        )
        
        # 5. Create a retriever with MMR for better diversity
        retriever = document_search.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "fetch_k": min(20, len(texts)),
                "lambda_mult": 0.5
            }
        )
        
        # 6. Create a prompt template for better answers
        template = """Use the following pieces of context to answer the question at the end. 
        If the context doesn't contain enough information to answer the question, 
        just say that you don't know based on the provided information.
        
        Context:
        {context}
        
        Question: {question}
        
        Provide a detailed and comprehensive answer based on the context above.
        Answer:"""
        
        QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=["context", "question"],
            template=template,
        )
        
        # 7. Create the QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
            return_source_documents=True
        )
        
        # 8. Get the answer
        st.info("Generating answer...")
        # Using invoke() instead of __call__ to avoid deprecation warning
        result = qa_chain.invoke({"query": query_text})
        
        # 9. Format the response with sources
        response = {
            "answer": result["result"],
            "sources": []
        }
        
        # Add source documents if available
        if result.get("source_documents"):
            for i, doc in enumerate(result["source_documents"], 1):
                response["sources"].append({
                    "id": i,
                    "page": doc.metadata.get("page", "N/A"),
                    "content": doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else "")
                })
        
        return response
        
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return f"Error: Could not generate a response. {str(e)}"

def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        if not text.strip():
            st.error("Could not extract any text from the PDF. The PDF might be scanned or protected.")
            return None
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {str(e)}")
        return None

def main():
    """Main function to run the Streamlit app."""
    # --- Streamlit Page Configuration ---
    st.set_page_config(
        page_title="Chat with your PDF (Local Version)",
        page_icon="💬",
        layout="wide"
    )
    
    st.title("Chat with Your Notes (100% Local) 💬")

    # Sidebar with instructions
    with st.sidebar:
        st.title("ℹ️ How to use")
        st.markdown("""
        1. Upload a PDF file
        2. Ask a question about the document
        3. Get instant answers!
        
        *No API keys needed. Everything runs locally on your machine.*
        *First run may take a few minutes to download the models.*
        """)
        
        st.markdown("---")
        st.markdown("### System Information")
        st.write(f"Python: {sys.version.split()[0]}")
        st.write(f"PyTorch: {torch.__version__}")
        st.write(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            st.write(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # File upload
    st.header("1. Upload your PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file", 
        type=["pdf"],
        label_visibility="collapsed"
    )
    
    st.header("2. Ask a question")
    question = st.text_area(
        "Enter your question about the document:",
        placeholder="What is this document about?",
        label_visibility="collapsed"
    )
    
    return uploaded_file, question

if __name__ == "__main__":
    # Get user inputs
    uploaded_file, question = main()
    
    # Add some spacing
    st.write("")
    
    # Generate response when button is clicked
    if st.button("Get Answer", type="primary", use_container_width=True):
        if not uploaded_file:
            st.error("Please upload a PDF file first!")
        elif not question.strip():
            st.error("Please enter a question!")
        else:
            with st.spinner("Processing your question..."):
                try:
                    response = generate_response(uploaded_file, question)
                    
                    if isinstance(response, str) and response.startswith("Error:"):
                        st.error(response)
                    else:
                        # Display the answer
                        st.markdown("### Answer")
                        st.markdown(response["answer"])
                        
                        # Display sources if available
                        if response["sources"]:
                            st.markdown("\n### Sources")
                            for source in response["sources"]:
                                with st.expander(f"Source {source['id']} (Page {source['page']})"):
                                    st.markdown(source['content'])
                        
                        # Add some spacing at the bottom
                        st.write("")
                        st.markdown("---")
                        st.caption("Note: This is a local AI model. No data was sent to any external servers.")
                        
                except Exception as e:
                    st.error(f"An error occurred while generating the response.")
                    st.exception(e)  # Show full traceback for debugging