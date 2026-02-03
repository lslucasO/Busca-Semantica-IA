import os, json, hashlib
from langchain_openai import OpenAI
from langchain_community.cache import InMemoryCache, SQLiteCache
from langchain_classic.globals import set_llm_cache
from dotenv import load_dotenv

load_dotenv()

openai = OpenAI(model_name='gpt-3.5-turbo-instruct', api_key=os.getenv('OPENAI_API_KEY'))


class SimpleDiskCache:
    def __init__(self, cache_dir='cache_dir'):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, key):
        hashed_key = hashlib.md5(key.encode()).hexdigest() #hasg cria nome de arquivo único
        return os.path.join(self.cache_dir, f"{hashed_key}.json")

    def lookup(self, key, llm_string):
        cache_path = self._get_cache_path(key)
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                return json.load(f)
        return None

    def update(self, key, value, llm_string):
        cache_path = self._get_cache_path(key)
        with open(cache_path, 'w') as f:
            json.dump(value, f)


cache = SimpleDiskCache()
set_llm_cache(cache)
prompt = 'Me diga quem foi Neil Degrasse Tyson.'


def invoke_with_cache(llm, prompt, cache):
    cached_response = cache.lookup(prompt, "")
    if cached_response:
        print("Usando cache:")
        return cached_response

    response = llm.invoke(prompt)
    cache.update(prompt, response, "")
    return response


response1 = invoke_with_cache(openai, prompt, cache)
response_text1 = response1.replace('\n', ' ') 

print("Primeira resposta (API chamada):", response_text1)

response2 = invoke_with_cache(openai, prompt, cache)
response_text2 = response2.replace('\n', ' ')  

print("Segunda resposta (usando cache):", response_text2)


# # Completion
# def in_cache():
#     # Grava na Memória
#     set_llm_cache(InMemoryCache())

#     prompt = 'Me diga em poucas palavras que foi Carl Sagan.'
#     response1 = openai.invoke(prompt)
#     print("Primeira resposta (API chamada):", response1)


#     response2 = openai.invoke(prompt)
#     print("Segunda resposta (usando cache):", response2)


# def in_sqlite():
#     set_llm_cache(SQLiteCache(database_path="openai_cache.db"))
    
#     prompt = 'Me diga com poucas palavras quem foi Neil Armstrong.'
#     response1 = openai.invoke(prompt)
    
#     print("Primeira resposta (API chamada):", response1)
    
#     response2 = openai.invoke(prompt)
#     print("Segunda resposta (usando cache):", response2)
    
# in_sqlite()