from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import streamlit as st
from streamlit_chat import message
from utils import *
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import DirectoryLoader
import emoji
from langchain.document_loaders import WebBaseLoader
import base64

#import { PromptTemplate } from "langchain/prompts"

general_system_template = r""" 
Your name is Anyware Product chatbot. 
Your primary goal is to help users with questions about Anyware Client, Anyware Agent and Anyware Manager.
Do not answer questions that are not related to Anyware Client or Anyware Agent or Anyware Manager and respond with I can only help with Anyware Endpoint /Anyware Manager related questions.
If you do not know the answer, respond with Sorry I do not know the answer to your question.  Please contact Teradici Support at https://help.teradici.com/s/
You were created by a team learning AI
You can only answer with information from https://en.wikipedia.org/wiki/NATO
 ----
{context}
----
"""
general_user_template = "Question:```{question}```"
messages = [
            SystemMessagePromptTemplate.from_template(general_system_template),
            HumanMessagePromptTemplate.from_template(general_user_template)
]
qa_prompt = ChatPromptTemplate.from_messages( messages )


title = emoji.emojize(":page_facing_up: :rainbow[Documentation Liberator Chatbot] :robot: :grinning_face_with_big_eyes: :nerd_face: :face_with_monocle:")
st.set_page_config(layout="wide")
st.image("ui/logo.png", width = 100)
st.header(title, divider='rainbow')

# get environemt variables
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
pinecone_index = os.getenv("PINECONE_INDEX")

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background("ui/books.jpg")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature = 0.2)

if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)

# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()

#Storing embeddings in Pinecone 
import pinecone 
from langchain.vectorstores import Pinecone
pinecone.init(
    api_key=pinecone_api_key,
    environment=pinecone_environment
)
index_name=pinecone_index
from langchain.embeddings import SentenceTransformerEmbeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Pinecone.from_existing_index(index_name, embeddings)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    vectordb.as_retriever(search_kwargs={'k': 2}),
    combine_docs_chain_kwargs={'prompt': qa_prompt},
    return_source_documents=True
)

import pprint
pp = pprint.PrettyPrinter(indent=4)

chat_history = []
with textcontainer:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("typing..."):
            conversation_string = get_conversation_string()
            # st.code(conversation_string)
            #refined_query = query_refiner(conversation_string, query)
            #st.subheader("Refined Query:")
            #st.write(refined_query)
            #context = find_match(refined_query)
            # print(context)  
            #response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
            result = qa_chain({"question": query, "chat_history": chat_history})
            chat_history.append((query, result['answer']))
            pp.pprint(result)
            citation = ""
            source_links = set()
            for src in result["source_documents"]:
                if src.metadata['source'] not in source_links:
                    source_links.add(src.metadata['source'])
                    citation += emoji.emojize("\n:right_arrow:" + "Source: " + src.metadata['title'] + "(" + ":link:" + src.metadata['source'] + ")\n")
            response = result['answer'] + "\n" + citation


        st.session_state.requests.append(query)
        st.session_state.responses.append(response) 
with response_container:
    if st.session_state['responses']:

        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')

          
