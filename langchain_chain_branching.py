import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableBranch
from langchain.schema.output_parser import StrOutputParser

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(model="gpt-3.5-turbo",max_completion_tokens=50)

chat_classfication_template = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful AI Assistant"), 
    ("human","Classify the input message into one of positive, negative, neutral, escalate {feedback}")]
)

positive_feedback_template = ChatPromptTemplate.from_messages(
    [("system","You are an AI Assistant"),
    ("human","Provide the response for the feedback in postive manner in less than 20 words {feedback}")]
)

negative_feedback_template = ChatPromptTemplate.from_messages(
    [("system","You are an AI Assistant"),
    ("human","Provide the response for the feedback in negative manner in less than 20 words {feedback}")]
)

neutral_feedback_template = ChatPromptTemplate.from_messages(
    [("system","You are an AI Assistant"),
    ("human","Provide the response for the feedback in neutral manner in less than 20 words {feedback}")]
)

escalate_feedback_template = ChatPromptTemplate.from_messages(
    [("system","You are an AI Assistant"),
    ("human","Provide the response for the feedback in escalate manner in less than 20 words {feedback}")]
)

branch = RunnableBranch(
    (lambda x: "positive" in x, positive_feedback_template | model | StrOutputParser()), 
    (lambda x: "negative" in x, negative_feedback_template | model | StrOutputParser()),
    (lambda x: "neutral" in x, neutral_feedback_template | model | StrOutputParser()),
    (escalate_feedback_template | model | StrOutputParser())
)

classification_chain = chat_classfication_template | model | StrOutputParser()

# Run the chain with an example review
# Good review - "The product is excellent. I really enjoyed using it and found it very helpful."
# Bad review - "The product is terrible. It broke after just one use and the quality is very poor."
# Neutral review - "The product is okay. It works as expected but nothing exceptional."
# Default - "I'm not sure about the product yet. Can you tell me more about its features and benefits?"

review = "I'm not sure about the product yet and I would like to escalate it"
chain = classification_chain | branch
response = chain.invoke({"feedback":review})
print(response)