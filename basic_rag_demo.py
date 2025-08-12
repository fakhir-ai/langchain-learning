from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
import os
from dotenv import load_dotenv 


from langchain.chains import RetrievalQA

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

loaders=TextLoader("data/textdata.txt")
document=loaders.load()

text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
texts=text_splitter.split_documents(document)

embeddings = OpenAIEmbeddings()

VectorStore=Chroma.from_documents(texts, embeddings)
qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model="gpt-3.5-turbo"), chain_type="stuff", retriever=VectorStore.as_retriever())

#query 
query = "What is interpreted Language?"
result = qa.invoke({"query": query})

print(result)
