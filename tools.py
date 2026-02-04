# Setup
from langchain_classic.utilities import PythonREPL
from langchain_community.tools import DuckDuckGoSearchRun,WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
# Executa Código Python
python_repl = PythonREPL()
result = python_repl.run("print(5 * 5)")
print(result)
# Busca na Web
ddg_search = DuckDuckGoSearchRun()

query = "Qual é a capital da Guiana?"
search_results = ddg_search.run(query)
print("Resultados da busca:", search_results)

# Consulta a Wikipedia
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
query = "Guiana"
wikipedia_results = wikipedia.run(query)
print("Resultados da Wikipedia:", wikipedia_results)
