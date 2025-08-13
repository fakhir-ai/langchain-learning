import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

model = init_chat_model("gpt-3.5-turbo", model_provider="openai")

messages = [SystemMessage("Translate the following message from English to Arabic"), 
            HumanMessage("How are you?")]

response = model.invoke(messages)
print(response.content)

system_template = "Translate the following message from English to {language}"
prompt_template = ChatPromptTemplate.from_messages([("system", system_template), ("user", "{text}")])

prompt = prompt_template.invoke({"language":"Arabic", "text":"How are you?"})
responsewithprompt = model.invoke(prompt)
print(responsewithprompt.content)
