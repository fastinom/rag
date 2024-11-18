import streamlit as st
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field

from helper_functions import encode_pdf  # Ensure you have these helper functions available
from evaluation.evalute_rag import *     # Import your evaluation logic

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Define relevant classes
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
        self.llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=1000, temperature=0)

        # Create LLMChains for each step
        self.retrieval_chain = retrieval_prompt | self.llm.with_structured_output(RetrievalResponse)
        self.relevance_chain = relevance_prompt | self.llm.with_structured_output(RelevanceResponse)
        self.generation_chain = generation_prompt | self.llm.with_structured_output(GenerationResponse)
        self.support_chain = support_prompt | self.llm.with_structured_output(SupportResponse)
        self.utility_chain = utility_prompt | self.llm.with_structured_output(UtilityResponse)

    def run(self, query):
        # Main logic for processing the query (use the same logic from your script)
        # Return the best response
        pass  # Adapt the run logic as necessary

# Streamlit App
st.title("SelfRAG: Retrieval-Augmented Generation")

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
    if uploaded_file and query:
        try:
            rag = SelfRAG(path="uploaded.pdf", top_k=top_k)
            response = rag.run(query)
            st.write("### Final Response")
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please upload a PDF and enter a query.")

# Notes
st.write("This app uses LangChain and OpenAI to process queries with Retrieval-Augmented Generation.")
