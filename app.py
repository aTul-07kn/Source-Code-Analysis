import os
from flask import Flask, jsonify, render_template, request
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
from src.helper import load_embedding
from src.helper import clone_repo
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain

app=Flask(__name__)

load_dotenv()
NVIDIA_API_KEY=os.getenv("NVIDIA_API_KEY")
os.environ["NVIDIA_API_KEY"]=NVIDIA_API_KEY

embedder=load_embedding()
vectordb = Chroma(embedding_function=embedder, persist_directory='./chromaDb')

llm = ChatNVIDIA(
  model="nvidia/llama-3.1-nemotron-70b-instruct",
  temperature=0.6,
  top_p=1,
  max_tokens=1024,
  stream=True
)

memory = ConversationSummaryMemory(llm=llm, memory_key = "chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(llm, retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k":8}), memory=memory)

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template('index.html')

@app.route("/store_index", methods=["GET", "POST"])
def gitRepo():
    if request.method=="POST":
        user_input=request.form["question"]
        clone_repo(user_input)
        os.system("python store_index.py")
    
    return jsonify({"response": str(user_input)})
    
@app.route("/chat", methods=["GET", "POST"])
def chat():
    input=request.form["msg"]
    print(input)
    
    if input=="clear":
        os.system("rm -rf repo")
    
    result=qa(input)
    print(result["answer"])
    return str(result["answer"])

if __name__=="__main__":
    app.run(port=8080, debug=True)