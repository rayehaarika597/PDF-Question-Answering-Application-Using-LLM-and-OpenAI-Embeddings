import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from PIL import Image
import openai

# Set your OpenAI API key directly


# Set up the Streamlit app configuration
img = Image.open(r"D:\DocGenius-Revolutionizing-PDFs-with-AI-main\DocGenius-Revolutionizing-PDFs-with-AI-main\images.jpeg")
st.set_page_config(page_title="DocGenius: Document Generation AI", page_icon=img)
st.header("Ask Your PDF📄")

# Upload PDF file
pdf = st.file_uploader("Upload your PDF", type="pdf")

if pdf is not None:
    # Read and extract text from the PDF
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Split the text into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Create embeddings and build the knowledge base
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    # Input for querying the knowledge base
    query = st.text_input("Ask your Question about your PDF")
    if query:
        docs = knowledge_base.similarity_search(query)

        # Load the question-answering chain and generate a response
        llm = OpenAI(api_key=openai_api_key)
        chain = load_qa_chain(llm, chain_type="stuff")
        try:
            response = chain.run(input_documents=docs, question=query)
            st.success(response)
        except openai.error.RateLimitError as e:
            st.error("Rate limit exceeded. Please try again later.")
else:
    st.info("Please upload a PDF file to get started.")
