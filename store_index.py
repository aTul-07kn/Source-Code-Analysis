from src.helper import clone_repo
from src.helper import load_repo
from src.helper import split_data
from src.helper import load_embedding
from langchain.vectorstores import Chroma

# clone_repo("https://github.com/aTul-07kn/Medi-Chatbot-Medicrolina")

docs=load_repo("repo/")

chunks=split_data(docs)

embedder=load_embedding()

vectordb = Chroma.from_documents(chunks, embedding=embedder, persist_directory='./chromaDb')

vectordb.persist()