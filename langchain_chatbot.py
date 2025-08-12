from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

st.title("Welcome to the ChatBot")
input_text = st.text_input("Hi! Have a wonderful time! Enter your question")

prompt = ChatPromptTemplate.from_messages([("system", "You are an AI Assistant Chatbot with polite manners"), 
                                  ("user", "user query : {query}")])

llm=ChatOpenAI(model="gpt-3.5-turbo")
output_parser = StrOutputParser()
chain = prompt|llm|output_parser 

if input_text:
    st.write(chain.invoke({"query":input_text}))
