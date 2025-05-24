import os
from git import Repo
from langchain.text_splitter import Language
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

NVIDIA_API_KEY=os.getenv("NVIDIA_API_KEY")
os.environ["NVIDIA_API_KEY"]=NVIDIA_API_KEY

def clone_repo(repo_url="https://github.com/aTul-07kn/Medi-Chatbot-Medicrolina"): 
    os.makedirs("repo", exist_ok=True)
    repo_path = "repo/"
    repo = Repo.clone_from(repo_url, to_path=repo_path)

def load_repo(repo_path):
    loader = GenericLoader.from_filesystem(repo_path,
                                        glob = "**/*",
                                        suffixes=[".py"],
                                        parser = LanguageParser(language=Language.PYTHON, parser_threshold=500))
    documents=loader.load()
    return documents

def split_data(documents):
    splitter=RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=500,
        chunk_overlap=50
    )
    
    chunks=splitter.split_documents(documents)
    return chunks

def load_embedding():
    embedder = NVIDIAEmbeddings(
    model="NV-Embed-QA",  
    truncate="NONE", 
    )
    
    return embedder