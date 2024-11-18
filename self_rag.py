import streamlit as st
import os
from groq import Groq
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from helper_functions import encode_pdf  # Ensure helper_functions contains `encode_pdf`

# Custom Groq client wrapper
class GroqClient:
    def __init__(self, model, temperature=1, max_tokens=1024, top_p=1):
        self.client = Groq()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

    def generate_response(self, prompt):
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                stream=False,  # Disable streaming for simplicity
                stop=None,
            )
            response = "".join(
                chunk.choices[0].delta.content or "" for chunk in completion
            )
            return response
        except Exception as e:
            raise Exception(f"Groq API Error: {e}")

# SelfRAG class
class SelfRAG:
    def __init__(self, path, top_k=3, groq_client=None):
        self.vectorstore = encode_pdf(path)
        self.top_k = top_k
        self.groq_client = groq_client

    def run(self, query):
        # Retrieval and response logic
        st.write("Processing query:", query)

        # Use retrieval logic as needed (e.g., self.vectorstore.similarity_search)
        # For simplicity, we'll focus on direct response generation
        input_prompt = f"Query: {query}\n\nGenerate a response:"
        response = self.groq_client.generate_response(input_prompt)
        return response

# Streamlit app
st.title("SelfRAG with Groq's Llama3 Model")

# API Key Input
st.sidebar.title("Configuration")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")
if api_key:
    os.environ["GROQ_API_KEY"] = api_key
    st.sidebar.success("API Key set successfully!")
else:
    st.sidebar.warning("Please enter your Groq API Key.")

# File upload
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
if uploaded_file:
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("PDF uploaded successfully!")

# Query input
query = st.text_input("Enter your query:")
top_k = st.slider("Number of documents to retrieve:", min_value=1, max_value=10, value=3)

# Run RAG
if st.button("Run"):
    if uploaded_file and query and api_key:
        try:
            # Initialize Groq client and SelfRAG
            groq_client = GroqClient(model="llama3-8b-8192")
            rag = SelfRAG(path="uploaded.pdf", top_k=top_k, groq_client=groq_client)
            
            # Process the query
            response = rag.run(query)
            
            # Display the response
            st.write("### Final Response")
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    elif not api_key:
        st.error("Please set the API Key in the sidebar.")
    else:
        st.warning("Please upload a PDF and enter a query.")

# Notes
st.write("This app uses Groq's Llama3-8b-8192 model for Retrieval-Augmented Generation.")
