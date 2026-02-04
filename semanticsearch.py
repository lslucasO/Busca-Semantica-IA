# Setup
import os, zipfile
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_openai import ChatOpenAI
from pinecone import Pinecone
from dotenv import load_dotenv


load_dotenv()


class SemanticSearch:
    def __init__(self, mode):
        """
        mode:
        - 'ingest' -> indexa documentos
        - 'query'  -> apenas consulta
        """
        self.mode = mode

        self.pinecone_client = Pinecone(
            api_key=os.environ["PINECONE_API_KEY"]
        )

        # IMPORTANTE: manter o mesmo modelo da ingestão
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002"
        )

        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.2
        )

        self.index_name = "llm"
        self.zip_file_path = "documentos.zip"
        self.extracted_folder_path = "docs"

        self.documents = []
        self.chunks = []


    # -------------------------
    # FLOW PRINCIPAL
    # -------------------------
    def run(self):
        if self.mode == "ingest":
            self.ingest()

        if self.mode == "query":
            self.query()


    # -------------------------
    # INGESTÃO (RODA UMA VEZ)
    # -------------------------
    def ingest(self):
        print("🔹 MODO INGESTÃO")

        self.extract_documents()
        self.load_documents()
        self.chunking()
        self.create_vector_store()

        print("✅ Indexação concluída")


    def extract_documents(self):
        with zipfile.ZipFile(self.zip_file_path, "r") as zip_ref:
            zip_ref.extractall(self.extracted_folder_path)


    def load_documents(self):
        for filename in os.listdir(self.extracted_folder_path):
            if filename.endswith(".pdf"):
                file_path = os.path.join(self.extracted_folder_path, filename)
                loader = PyMuPDFLoader(file_path)
                self.documents.extend(loader.load())


    def chunking(self):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
        )

        self.chunks = splitter.create_documents(
            [doc.page_content for doc in self.documents]
        )


    def create_vector_store(self):
        PineconeVectorStore.from_documents(
            documents=self.chunks,
            embedding=self.embeddings,
            index_name=self.index_name
        )


    # -------------------------
    # CONSULTA (RODA SEMPRE)
    # -------------------------
    def query(self):
        print("🔹 MODO CONSULTA")

        # conecta no cliente Pinecone
        index = self.pinecone_client.Index(self.index_name)

        # conecta o LangChain ao índice EXISTENTE
        vector_store = PineconeVectorStore(
            index=index,
            embedding=self.embeddings
        )

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever
        )

        for q in self.get_queries():
            response = chain.invoke(q)
            print(f"\nPrompt: {q}")
            print(f"Resposta: {response['result']}")



    def get_queries(self):
        return [
            "Qual o número do processo que trata de Violação de normas ambientais pela Empresa de Construção?",
            "Qual foi a decisão no caso de fraude financeira?",
            "Quais foram as alegações no caso de negligência médica?",
            "Quais foram as alegações no caso de Número do Processo: 822162?"
        ]


# -------------------------
# EXECUÇÃO
# -------------------------

# USE UMA VEZ:
# semantic = SemanticSearch(mode="ingest")

# USE SEMPRE:
semantic = SemanticSearch(mode="query")
semantic.run()
