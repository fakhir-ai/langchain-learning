import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda

load_dotenv()
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

message = [("system","What is {topic}"), "human","explain it in {count} points"]
model = ChatOpenAI(model='gpt-3.5-turbo',temperature=0.6)
prompt = ChatPromptTemplate.from_messages(message)

format_upper = RunnableLambda(lambda x : x.upper())
count_words = RunnableLambda(lambda x : f"Word Count :  {len(x.split())}\n{x}")

chain = prompt | model | StrOutputParser() | format_upper | count_words
response = chain.invoke({"topic":"Generative AI","count":3})

print(response)

