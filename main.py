import os
from urllib import response
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def complete_sentence(prompt):
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=50
    )
     
    print(response.choices[0].text.strip())
   
    
def complete_chat_sentence(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=50,
    )
     
    print(response.choices[0].message.content.strip())


complete_chat_sentence("Complete the following sentence: The capital of France is")
