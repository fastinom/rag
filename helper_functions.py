from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain import PromptTemplate
from typing import List
from rank_bm25 import BM25Okapi
import fitz
import asyncio
import random
import textwrap
import numpy as np
from enum import Enum
from groq import Groq

# Initialize Groq client
groq_client = Groq()


def replace_t_with_space(list_of_documents):
    """
    Replaces all tab characters ('\t') with spaces in the page content of each document.
    """
    for doc in list_of_documents:
        doc.page_content = doc.page_content.replace('\t', ' ')
    return list_of_documents


def text_wrap(text, width=120):
    """
    Wraps the input text to the specified width.
    """
    return textwrap.fill(text, width=width)


def get_groq_embeddings(texts):
    """
    Fetches embeddings for given texts using the Groq API.
    """
    embeddings = []
    for text in texts:
        response = groq_client.embeddings.create(model="llama3-8b-8192", input=text)
        embeddings.append(response["embedding"])
    return embeddings


def encode_pdf(path, chunk_size=1000, chunk_overlap=200):
    """
    Encodes a PDF book into a vector store using Groq embeddings.
    """
    loader = PyPDFLoader(path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    texts = text_splitter.split_documents(documents)
    cleaned_texts = replace_t_with_space(texts)

    # Generate Groq embeddings
    text_contents = [doc.page_content for doc in cleaned_texts]
    embeddings = get_groq_embeddings(text_contents)

    # Create vector store
    vectorstore = FAISS.from_texts(text_contents, embeddings)
    return vectorstore


def encode_from_string(content, chunk_size=1000, chunk_overlap=200):
    """
    Encodes a string into a vector store using Groq embeddings.
    """
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Content must be a non-empty string.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    chunks = text_splitter.create_documents([content])
    cleaned_chunks = replace_t_with_space(chunks)

    text_contents = [chunk.page_content for chunk in cleaned_chunks]
    embeddings = get_groq_embeddings(text_contents)

    vectorstore = FAISS.from_texts(text_contents, embeddings)
    return vectorstore


def retrieve_context_per_question(question, chunks_query_retriever):
    """
    Retrieves relevant context for a question using the chunks query retriever.
    """
    docs = chunks_query_retriever.get_relevant_documents(question)
    context = [doc.page_content for doc in docs]
    return context


class QuestionAnswerFromContext(BaseModel):
    answer_based_on_content: str = Field(description="Generates an answer to a query based on a given context.")


def create_question_answer_from_context_chain():
    """
    Sets up a question-answer chain using Groq completions.
    """
    question_answer_prompt_template = """ 
    For the question below, provide a concise but suffice answer based ONLY on the provided context:
    {context}
    Question
    {question}
    """

    question_answer_from_context_prompt = PromptTemplate(
        template=question_answer_prompt_template,
        input_variables=["context", "question"],
    )

    def groq_completion(context, question):
        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": question}
            ],
            temperature=0.7,
            max_tokens=150
        )
        return response["choices"][0]["message"]["content"]

    return question_answer_from_context_prompt | groq_completion


def answer_question_from_context(question, context, question_answer_from_context_chain):
    """
    Answer a question using Groq completions and the provided context.
    """
    input_data = {"question": question, "context": context}
    output = question_answer_from_context_chain.invoke(input_data)
    answer = output.answer_based_on_content
    return {"answer": answer, "context": context, "question": question}


def read_pdf_to_string(path):
    """
    Reads a PDF and returns its content as a string.
    """
    doc = fitz.open(path)
    content = ""
    for page_num in range(len(doc)):
        page = doc[page_num]
        content += page.get_text()
    return content


def bm25_retrieval(bm25: BM25Okapi, cleaned_texts: List[str], query: str, k: int = 5):
    """
    Perform BM25 retrieval and return the top k cleaned text chunks.
    """
    query_tokens = query.split()
    bm25_scores = bm25.get_scores(query_tokens)
    top_k_indices = np.argsort(bm25_scores)[::-1][:k]
    top_k_texts = [cleaned_texts[i] for i in top_k_indices]
    return top_k_texts
