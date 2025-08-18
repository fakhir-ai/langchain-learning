import os
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSerializable
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableParallel, RunnableLambda
from langchain.schema.output_parser import StrOutputParser

load_dotenv()
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(model="gpt-3.5-turbo",max_completion_tokens=50)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are the movie reviewer"), 
    ("human","Provide the moview review for the {movie}")
])

def pros_review(movies):
    pros_template = ChatPromptTemplate.from_messages([("system","You are the {movies} reviewer"), ("human","For the {movies}, provide the positive review")])
    return pros_template.format_prompt(movies=movies) 

def cons_review(movies):
    cons_template = ChatPromptTemplate.from_messages([("system","You are the {movies} reviewer"),
    ("human","For the {movies}, provide the negative review")])
    return cons_template.format_prompt(movies=movies)

analyze_pros_chain = RunnableLambda(lambda x: pros_review(x)) | model | StrOutputParser()

analyze_cons_chain = RunnableLambda(lambda x: cons_review(x)) | model | StrOutputParser()

def combine_pros_cons_chain(pros,cons):
    return f"Pros:\n {pros}\n\n Cons:\n {cons}" 

chain = (prompt_template | model | StrOutputParser() |
        RunnableParallel(branches={"pros":analyze_pros_chain,"cons":analyze_cons_chain})|
        RunnableLambda(lambda x: combine_pros_cons_chain(x["branches"]["pros"], x["branches"]["cons"])) 
)

response = chain.invoke({"movie":"Jurassic World"})
print(response)