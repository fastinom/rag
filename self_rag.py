import streamlit as st
from helper_functions import (
    encode_pdf,
    encode_from_string,
    answer_question_from_context,
    read_pdf_to_string,
    create_question_answer_from_context_chain,
    bm25_retrieval
)
from rank_bm25 import BM25Okapi

# Streamlit Page Setup
st.set_page_config(page_title="PDF Summarizer with Q&A", layout="wide")
st.title("PDF Summarizer and Q&A with Groq")

# State Variables
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None

if "bm25" not in st.session_state:
    st.session_state["bm25"] = None

if "cleaned_texts" not in st.session_state:
    st.session_state["cleaned_texts"] = None

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

# PDF Processing
if uploaded_file:
    st.write("Processing the PDF...")
    pdf_path = uploaded_file.name
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    # Extract text from PDF
    pdf_content = read_pdf_to_string(pdf_path)

    # Encode the PDF into vector store
    with st.spinner("Encoding PDF into vector store..."):
        st.session_state["vectorstore"] = encode_pdf(pdf_path)
        st.success("Vector store created!")

    # Prepare for BM25 retrieval
    with st.spinner("Preparing BM25 retriever..."):
        chunk_size = 1000
        chunk_overlap = 200
        st.session_state["bm25"] = BM25Okapi(pdf_content.split())
        st.session_state["cleaned_texts"] = pdf_content.split()

# Summarize PDF
if st.session_state["vectorstore"]:
    st.header("PDF Summarization")

    # Summarize entire PDF or a given section
    summarization_prompt = st.text_area(
        "Enter a summarization prompt (optional):",
        value="Summarize the uploaded document."
    )
    if st.button("Summarize"):
        chain = create_question_answer_from_context_chain()
        context = st.session_state["vectorstore"].similarity_search(summarization_prompt, k=5)
        context_text = " ".join([doc.page_content for doc in context])
        answer = answer_question_from_context(
            summarization_prompt, context_text, chain
        )
        st.write("**Summary:**")
        st.write(answer["answer"])

# Question-Answering
if st.session_state["vectorstore"]:
    st.header("Question Answering")

    # Enter question
    question = st.text_input("Ask a question about the document:")

    if st.button("Get Answer"):
        chain = create_question_answer_from_context_chain()
        # Retrieve context for the question
        with st.spinner("Retrieving context..."):
            context = st.session_state["vectorstore"].similarity_search(question, k=5)
            context_text = " ".join([doc.page_content for doc in context])

        # Generate answer using Groq
        with st.spinner("Generating answer..."):
            answer = answer_question_from_context(question, context_text, chain)
            st.write("**Answer:**")
            st.write(answer["answer"])
            st.write("**Context:**")
            st.write(answer["context"])

# BM25 Search
if st.session_state["bm25"]:
    st.header("BM25 Context Retrieval")

    bm25_query = st.text_input("Enter a query for BM25 search:")

    if st.button("Retrieve BM25 Context"):
        with st.spinner("Searching with BM25..."):
            bm25_results = bm25_retrieval(
                st.session_state["bm25"], st.session_state["cleaned_texts"], bm25_query
            )
            st.write("**Top Contexts:**")
            for i, text in enumerate(bm25_results):
                st.write(f"**Chunk {i + 1}:**")
                st.write(text)
