import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage


load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, max_completion_tokens=30)

# Single template message
messages = "Tell me about {topic}"
template = ChatPromptTemplate.from_template(messages)
prompt_template = template.invoke({"topic":"chatgpt"})
response = model.invoke(prompt_template)
print(response.content)

# # Multi-Template Message
messages = """ What does {profession} do? And Tell them in {count} lines"""
template = ChatPromptTemplate.from_template(messages)
prompt_template = template.invoke({"profession":"doctor","count":"3"}) 
response = model.invoke(prompt_template)
print(response.content)

# Template as Tuple
messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    HumanMessage(content="Tell me 3 jokes."),
]
template = ChatPromptTemplate.from_messages(messages)
prompt_template = template.invoke({"topic":"AI Engineer"})
response = model.invoke(prompt_template)
print(response.content)