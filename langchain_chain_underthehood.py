import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableSequence, RunnableLambda

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(model="gpt-3.5-turbo")
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an AI Travel Agent, Suggest travel places around {city}"), 
    ("human", "Share the top {count} tourist places" )])

format_template = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
model_invoke = RunnableLambda(lambda x: model.invoke(x.to_messages()))
output_format = RunnableLambda(lambda x: x.content )

chain = RunnableSequence(first=format_template,middle =[model_invoke],last=output_format)

response = chain.invoke({"city":"New York","count":"3"})
print(response)