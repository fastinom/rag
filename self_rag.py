import streamlit as st
from groq import Groq
import fitz  # PyMuPDF for PDF parsing
from typing import List

# Initialize the Groq client
client = Groq()

# Function to extract text from uploaded PDFs
def extract_text_from_pdfs(uploaded_files) -> List[str]:
    documents = []
    for uploaded_file in uploaded_files:
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text()
            documents.append(text)
    return documents

# Function to retrieve relevant context from PDFs
def retrieve_context(query: str, documents: List[str]) -> List[str]:
    # Simplified retrieval: search for query keywords in documents
    relevant_context = []
    for doc in documents:
        if query.lower() in doc.lower():
            relevant_context.append(doc[:500])  # Extract the first 500 characters as context
    return relevant_context

# Function to generate response using Groq's API
def generate_response(context: str, query: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query},
        {"role": "assistant", "content": f"Relevant context: {context}"},
    ]

    # Call the Groq API
    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=messages,
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=False,  # Set to False for single response
        stop=None,
    )
    
    # Access the content of the response
    response = completion.choices[0].message.content
    return response

# Streamlit App
st.title("RAG System with PDF Upload and Groq's Llama Model")
st.write("Upload PDFs, retrieve context, and generate responses using Groq's API.")

# Upload PDF files
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    # Extract text from PDFs
    st.write("### Extracting text from uploaded PDFs...")
    documents = extract_text_from_pdfs(uploaded_files)
    st.write(f"Extracted text from {len(documents)} document(s).")

    # Input query
    query = st.text_input("Enter your query:", "")

    if query:
        # Retrieve context
        st.write("### Retrieving relevant context...")
        context = retrieve_context(query, documents)
        st.write("#### Retrieved Context:")
        for i, ctx in enumerate(context, 1):
            st.write(f"{i}. {ctx}")

        if context:
            # Generate response
            st.write("### Generating response...")
            combined_context = "\n".join(context)
            response = generate_response(combined_context, query)

            # Display the response
            st.write("#### Response:")
            st.write(response)
        else:
            st.write("No relevant context found in the uploaded documents.")
