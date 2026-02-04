# Setup
from langchain_openai import OpenAI
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.prompts import PromptTemplate
import pandas as pd
import yaml
import os

with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)
os.environ['OPENAI_API_KEY'] = config['OPENAI_API_KEY']
openai = OpenAI(model_name='gpt-3.5-turbo-instruct', temperature=0)

# Agente de Busca e Resumo
ddg_search = DuckDuckGoSearchRun()
agent_executor = create_python_agent(
    llm=openai,
    tool=ddg_search,  
    verbose=True
)

prompt_template = PromptTemplate(
    input_variables=["query"],
    template="""
    Pesquise na web sobre {query} e forneça um resumo abrangente sobre o assunto.
    """
)

query = "Carl Sagan"
prompt = prompt_template.format(query=query)
print(prompt)
response = agent_executor.invoke(prompt)
print("Entrada do agente:", response['input'])
print("Saída do agente:", response['output'])
