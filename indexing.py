# Loading documents from a directory with LangChain
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import WebBaseLoader

import os

directory = 'data'

web_links = ["https://en.wikipedia.org/wiki/NATO"]

# get environemt variables
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
pinecone_index = os.getenv("PINECONE_INDEX")

def load_docs(directory):
  loader = WebBaseLoader(web_links)
  documents = loader.load()
  return documents

documents = load_docs(directory)

# Splitting documents
from langchain.text_splitter import RecursiveCharacterTextSplitter
def split_docs(documents,chunk_size=1000,chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
  texts = text_splitter.split_documents(documents)
  return texts

docs = split_docs(documents)

# Creating embeddings
from langchain.embeddings import SentenceTransformerEmbeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

query_result = embeddings.embed_query("Hello world")

#Storing embeddings in Pinecone 
import pinecone 
from langchain.vectorstores import Pinecone
pinecone.init(
    api_key=pinecone_api_key,
    environment=pinecone_environment
)
index_name = pinecone_index
index = Pinecone.from_documents(docs, embeddings, index_name=index_name)
