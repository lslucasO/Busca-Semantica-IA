# Setup
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_openai import ChatOpenAI
from pinecone import Pinecone
import yaml
import os
import zipfile
from dotenv import load_dotenv

load_dotenv()

pinecone_client = Pinecone(api_key=os.environ['PINECONE_API_KEY'])

# Busca Semântica
zip_file_path = 'documentos.zip'
extracted_folder_path = 'docs'


with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_folder_path)


documents = []
for filename in os.listdir(extracted_folder_path):
    if filename.endswith(".pdf"):
        file_path = os.path.join(extracted_folder_path, filename)
        loader = PyMuPDFLoader(file_path)
        documents.extend(loader.load())
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  
    chunk_overlap=100,
    length_function=len
)

chunks = text_splitter.create_documents([doc.page_content for doc in documents])
embeddings = OpenAIEmbeddings(model='text-embedding-ada-002') 
index_name = 'llm' 
vector_store = PineconeVectorStore.from_documents(chunks, embeddings, index_name=index_name)

query_1 = '''Responda apenas com base no input fornecido. Qual o número do processo que trata de Violação
de normas ambientais pela Empresa de Construção?'''
query_2 = 'Responda apenas com base no input fornecido. Qual foi a decisão no caso de fraude financeira?'
query_3 = 'Responda apenas com base no input fornecido. Quais foram as alegações no caso de negligência médica?'
query_4 = 'Responda apenas com base no input fornecido. Quais foram as alegações no caso de Número do Processo: 822162' #disputa contratual

llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.2)
retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})
chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)
print(chain)

answer_1 = chain.invoke(query_1)
answer_2 = chain.invoke(query_2)
answer_3 = chain.invoke(query_3)
answer_4 = chain.invoke(query_4)
print('Pergunta: ',answer_1['query'])
print('Resultado: ',answer_1['result'],'\n')
#---
print('Pergunta: ',answer_2['query'])
print('Resultado: ',answer_2['result'],'\n')
#---
print('Pergunta: ',answer_3['query'])
print('Resultado: ',answer_3['result'],'\n')
#---
print('Pergunta: ',answer_4['query'])
print('Resultado: ',answer_4['result'])
