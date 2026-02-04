# Setup
import os, zipfile
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_openai import ChatOpenAI
from pinecone import Pinecone
from dotenv import load_dotenv


load_dotenv()


class SemanticSearch:
    def __init__(self):
        self.pinecone_client = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
        self.embeddings = OpenAIEmbeddings(model='text-embedding-ada-002') 
        self.llm = ChatOpenAI(model_name='gpt-4o-mini', temperature=0.2)
        self.pinecone_index_name = 'llm'
        self.zip_file_path = 'documentos.zip'
        self.extracted_folder_path = 'docs'
        self.index_name = 'llm' 
        self.documents = []
    
    
    def main(self):
        self.extract_documents()
        self.insert_documents(self.extracted_folder_path)
        self.chunking()
        vector_store = self.vector_store()
        retriever = self.vector_retriever(vector_store)
        self.load_chain(retriever)
        queries = self.querys()
        self.answer_queries(queries)


    def extract_documents(self) -> None:
        with zipfile.ZipFile(self.zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(self.extracted_folder_path)

        
    def insert_documents(self, file_path) -> None:
        for filename in os.listdir(self.extracted_folder_path):
            if filename.endswith(".pdf"):
                file_path = os.path.join(self.extracted_folder_path, filename)
                loader = PyMuPDFLoader(file_path)
                self.documents.extend(loader.load())
          
            
    def chunking(self) -> None:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  
            chunk_overlap=100,
            length_function=len
        )
        
        self.chunks = text_splitter.create_documents(
            [doc.page_content for doc in self.documents]
        )

    
    def vector_store(self):
        vector_store = PineconeVectorStore.from_documents(
            self.chunks,
            self.embeddings,
            index_name=self.index_name
        )
        
        return vector_store


    def querys(self):
        query_1 = 'Responda apenas com base no input fornecido. Qual o número do processo que trata de Violação de normas ambientais pela Empresa de Construção?'
        query_2 = 'Responda apenas com base no input fornecido. Qual foi a decisão no caso de fraude financeira?'
        query_3 = 'Responda apenas com base no input fornecido. Quais foram as alegações no caso de negligência médica?'
        query_4 = 'Responda apenas com base no input fornecido. Quais foram as alegações no caso de Número do Processo: 822162' #disputa contratual

        return query_1, query_2, query_3, query_4


    def vector_retriever(self, vector_store):
        self.retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})
        
        return self.retriever
    
    
    def load_chain(self, retriever):
        self.chain = RetrievalQA.from_chain_type(llm=self.llm, chain_type='stuff', retriever=retriever)


    def answer_queries(self, queries):
        for query in queries:
            response = self.chain.invoke(query)
            print(f'Prompt: {query}\nResponse: {response["result"]}\n')
            

semantic_search = SemanticSearch()
semantic_search.main()






