import os
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from sqlalchemy.orm import query_expression


load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

system_message = [(SystemMessage("I am an AI Assistant speciailize in Maths, Do you have any question for me?"))]
model = ChatOpenAI(model='gpt-3.5-turbo',temperature=0)

chat_history = []
chat_history.append(system_message)

while True:
    query = input("You : ")
    if query.lower() == 'exit':
        print("exiting")
        break
    chat_history.append(HumanMessage(content=query))
    response = model.invoke(query)
    result = response.content
    chat_history.append(AIMessage(result))
    print(f"AI : {result}")
print(chat_history)
