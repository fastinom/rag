import streamlit as st
import os
import requests
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from helper_functions import encode_pdf  # Ensure these helper functions are correctly implemented

# Custom ChatOpenAI for Groq API compatibility
class CustomChatOpenAI:
    def __init__(self, model, max_tokens=1000, temperature=0):
        self.api_base = "https://api.groq.com/v1"  # Replace with the correct Groq endpoint
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Custom headers for Groq
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def call_api(self, prompt):
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        response = requests.post(
            f"{self.api_base}/chat/completions",
            headers=self.headers,
            json=payload,
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(
                f"API Error: {response.status_code}, {response.json().get('error')}"
            )


# Define relevant classes for LangChain
class RetrievalResponse(BaseModel):
    response: str = Field(..., title="Determines if retrieval is necessary", description="Output only 'Yes' or 'No'.")

class RelevanceResponse(BaseModel):
    response: str = Field(..., title="Determines if context is relevant",
                          description="Output only 'Relevant' or 'Irrelevant'.")

class GenerationResponse(BaseModel):
    response: str = Field(..., title="Generated response", description="The generated response.")

class SupportResponse(BaseModel):
    response: str = Field(..., title="Determines if response is supported",
                          description="Output 'Fully supported', 'Partially supported', or 'No support'.")

class UtilityResponse(BaseModel):
    response: int = Field(..., title="Utility rating", description="Rate the utility of the response from 1 to 5.")

# Prompt templates
retrieval_prompt = PromptTemplate(
    input_variables=["query"],
    template="Given the query '{query}', determine if retrieval is necessary. Output only 'Yes' or 'No'."
)

relevance_prompt = PromptTemplate(
    input_variables=["query", "context"],
    template="Given the query '{query}' and the context '{context}', determine if the context is relevant. Output only 'Relevant' or 'Irrelevant'."
)

generation_prompt = PromptTemplate(
    input_variables=["query", "context"],
    template="Given the query '{query}' and the context '{context}', generate a response."
)

support_prompt = PromptTemplate(
    input_variables=["response", "context"],
    template="Given the response '{response}' and the context '{context}', determine if the response is supported by the context. Output 'Fully supported', 'Partially supported', or 'No support'."
)

utility_prompt = PromptTemplate(
    input_variables=["query", "response"],
    template="Given the query '{query}' and the response '{response}', rate the utility of the response from 1 to 5."
)

# SelfRAG class
class SelfRAG:
    def __init__(self, path, top_k=3):
        self.vectorstore = encode_pdf(path)
        self.top_k = top_k
        self.llm = CustomChatOpenAI(model="gpt-4o-mini")

    def run(self, query):
        # Main logic for processing the query (adapted from your script)
        print(f"Processing query: {query}")

        # Implement retrieval, relevance, and generation logic
        input_prompt = f"Query: {query}\n\nGenerate a response:"
        response = self.llm.call_api(input_prompt)
        return response


# Streamlit App
st.title("SelfRAG: Retrieval-Augmented Generation")

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
            rag = SelfRAG(path="uploaded.pdf", top_k=top_k)
            response = rag.run(query)
            st.write("### Final Response")
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    elif not api_key:
        st.error("Please set the API Key in the sidebar.")
    else:
        st.warning("Please upload a PDF and enter a query.")

# Notes
st.write("This app uses Groq's API to process queries with Retrieval-Augmented Generation.")
