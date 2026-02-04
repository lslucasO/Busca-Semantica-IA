# Setup
from langchain.prompts import PromptTemplate
from langchain import hub
from langchain.agents import Tool, AgentExecutor, initialize_agent, create_react_agent
from langchain.tools import DuckDuckGoSearchRun
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_openai import ChatOpenAI
import yaml
import os
with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)
os.environ['OPENAI_API_KEY'] = config['OPENAI_API_KEY']
openai = ChatOpenAI(model_name='gpt-4-turbo-preview', temperature=0)
# Assitente Financeiro Pessoal
prompt = '''
Como assistente financeiro pessoal, ajude a responder as seguintes perguntas com ajuda da internet.
Perguntas: {q}
'''
prompt_template = PromptTemplate.from_template(prompt)
print(prompt_template)
react_instructions  = hub.pull('hwchase17/react')
print(react_instructions )
# Ferramenta 1 Python REPL
python_repl = PythonREPLTool()
python_repl_tool = Tool(
    name='Python REPL',
    func=python_repl.run,
    description='''Qualquer tipo de cálculo deve usar esta ferramenta. Você não deve realizar
                    o cálculo diretamente. Você deve inserir código Python.'''
)
# Ferramenta  2 busca DuckDuckGo 
search = DuckDuckGoSearchRun()
duckduckgo_tool = Tool(
    name='Busca DuckDuckGo',
    func=search.run,
    description='''Útil para encontrar informações e dicas de economia e opções de investimento.
                Você sempre deve pesquisar na internet as melhores dicas usando esta ferramenta, não
                responda diretamente. Sua resposta deve informar que há elementos pesquisados na internet'''
)
tools = [python_repl_tool, duckduckgo_tool]
agent = create_react_agent(openai, tools, react_instructions)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10
)

print(agent_executor)
question = """
Eu ganho R$4000 por mês mas o total de minhas despesas é de R$3800 mais 500 de aluguel.
Como posso ajustar meu orçamento para economizar dinheiro?
"""
question = """
Minha renda é de R$6000 por mês, o total de minhas despesas é de R$2500 mais 1000 de aluguel.
Quais dicas de investimento você me dá?
"""
output = agent_executor.invoke({
    'input': prompt_template.format(q=question)
})

print(output['input'])
print(output['output'])
