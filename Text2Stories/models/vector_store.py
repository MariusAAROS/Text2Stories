from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

import subprocess
import os

root = subprocess.run(["git", "rev-parse", "--show-toplevel"], 
capture_output=True, 
text=True,
encoding="utf-8").stdout.strip()

raw_path = os.path.join(root, "Data", "Raw")
directory = DirectoryLoader(raw_path).load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

docs = text_splitter.split_documents(directory)

embedder = OllamaEmbeddings(
    model="qwen2.5:3b"
)

vector_store = FAISS.from_documents(docs, embedder)

vector_store.save_local(folder_path="Knowledge/CM")