import os
from langchain_openai import ChatOpenAI, OpenAI
from dotenv import load_dotenv

load_dotenv()

# Completion client
def model_completion():
    openai = OpenAI(
    model="gpt-3.5-turbo-instruct",
    api_key=os.environ.get("OPENAI_API_KEY")
    )    
     
    response = openai.invoke(
    input="Complete the following sentence: The capital of France is",
    max_tokens=50,
    temperature=0.3,
    frequency_penalty=1,
    presence_penalty=1,
    seed=123
    )

    print(response.text.strip())
    

def model_chat():
    chat = ChatOpenAI(
    model="gpt-3.5-turbo",
    api_key=os.environ.get("OPENAI_API_KEY")
    )    
    
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Complete the following sentence: The capital of France is"},
    ]
    
    response = chat.invoke(messages)
    
    print(response.content.strip())
    
model_chat()