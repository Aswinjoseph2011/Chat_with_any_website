import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
genai.configure(api_key=google_api_key)

system_template = """Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}

def main():
    # Set the title and subtitle of the app
    #st.image("test.png", width=120)  # Adjust the path to your logo image file

    st.title('🔍 Chat With Website')
    st.subheader('Input your website URL, ask questions, and receive answers directly from the website.')

    url = st.text_input("Insert The website URL")

    prompt_text = st.text_input("Ask a question (query/prompt)")
    if st.button("Submit Query", type="primary"):
        ABS_PATH = os.path.dirname(os.path.abspath(__file__))
        DB_DIR = os.path.join(ABS_PATH, "db")

        # Load data from the specified URL
        loader = WebBaseLoader(url)
        data = loader.load()

        # Split the loaded data
        text_splitter = CharacterTextSplitter(separator='\n', 
                                              chunk_size=500, 
                                              chunk_overlap=40)
        docs = text_splitter.split_documents(data)

        # Create Google Generative AI embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Create a Chroma vector database from the documents
        vectordb = Chroma.from_documents(documents=docs, 
                                         embedding=embeddings,
                                         persist_directory=DB_DIR)
        vectordb.persist()

        # Create a retriever from the Chroma vector database
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})

        # Use a Google Generative AI model
        llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        #print(type(llm))

        # Create a RetrievalQA from the model and retriever
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        print(qa)
        # Run the prompt and return the response
        response = qa(prompt_text)
        st.write(response)

if __name__ == '__main__':
    main()
