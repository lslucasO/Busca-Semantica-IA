import os, json, hashlib
from langchain_openai import ChatOpenAI,OpenAI
from langchain_classic.prompts import PromptTemplate,ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

template = '''
Você é um analista financeiro.
Escreva um relatório financeiro detalhado para a empresa "{empresa}" para o período {periodo}.

O relatório deve ser escrito em {idioma} e incluir as seguintes análises:
{analises}

Certifique-se de fornecer insights e conclusões para cada seção.
'''

prompt_template = PromptTemplate.from_template(template=template)

empresa = 'ACME Corp'
periodo = 'Q1 2024'
idioma = 'Português'
analises = [
    "Análise do Balanço Patrimonial",
    "Análise do Fluxo de Caixa",
    "Análise de Tendências",
    "Análise de Receita e Lucro",
    "Análise de Posição de Mercado"
]
analises_text = "\n".join([f"- {analise}" for analise in analises])

prompt_template = PromptTemplate.from_template(template=template)

empresa = 'ACME Corp'
periodo = 'Q1 2024'
idioma = 'Português'
analises = [
    "Análise do Balanço Patrimonial",
    "Análise do Fluxo de Caixa",
    "Análise de Tendências",
    "Análise de Receita e Lucro",
    "Análise de Posição de Mercado"
]
analises_text = "\n".join([f"- {analise}" for analise in analises])
print(analises_text)

prompt = prompt_template.format(
    empresa=empresa,
    periodo=periodo,
    idioma=idioma,
    analises=analises_text
)
print("Prompt Gerado:\n", prompt)


openai = OpenAI(model_name='gpt-3.5-turbo-instruct',max_tokens=1000) #max 4096

response = openai.invoke(prompt)
print("Saída do LLM:\n", response)