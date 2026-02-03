# Setup
import numpy as np
from pinecone import Pinecone
import os 
from dotenv import load_dotenv
load_dotenv()

pinecone_client = Pinecone(api_key=os.environ['PINECONE_API_KEY'])

## Pinecone
indices = pinecone_client.list_indexes()


for index in indices:
    index_name = index['name']
    print(f"Index: {index_name}")
    print(pinecone_client.describe_index(index_name))
    
    
indice_nome = 'nlp'

vetores = [np.random.normal(0, 1, 2048).tolist() for _ in range(5)]
ids = ['a', 'b', 'c', 'd', 'e']

indice = pinecone_client.Index(indice_nome)
indice.upsert(vectors=list(zip(ids, vetores)))

print(len(vetores))
print(vetores[4])
print(indice.fetch(ids=['c']))

#1.29802501
response = indice.fetch(ids=['c'])
if 'vectors' in response and 'c' in response['vectors']:
    retorna_vetor = response['vectors']['c']['values']
    atualiza_vetor = [x + 1 for x in retorna_vetor]  
    indice.upsert(vectors=[('c', atualiza_vetor)])
    print(indice.fetch(ids=['c']))
else:
    print("Vetor com ID 'c' não encontrado no índice.")
    
indice.delete(ids=['d', 'e'])
print(indice.fetch(ids=['d', 'e'])) 
print(indice.describe_index_stats())

indice.upsert(vectors=list(zip(ids, vetores)), namespace='namespace1')
indice.upsert(vectors=list(zip(['x', 'y', 'z'], 
                               [np.random.normal(0, 1, 2048).tolist() for _ in range(3)])), 
                                namespace='namespace2')

print(indice.fetch(ids=['a'], namespace='namespace1'))
print(indice.fetch(ids=['x'], namespace='namespace2'))

indice.delete(ids=['x'], namespace='namespace2')

print(indice.fetch(ids=['x'], namespace='namespace2'))  # Deve retornar vazio
query_vector = np.random.normal(0, 1, 2048).tolist()

print(indice.query(vector=query_vector, top_k=3, include_values=False))
