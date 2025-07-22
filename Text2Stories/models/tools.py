from typing import Annotated

from langchain_core.tools import tool

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

@tool
def retriever(q: Annotated[str, "Query to compare with the vector store information"], 
                      k: Annotated[int, "Number of results to return from the similarity search"]):
    """
    Search the vector store for the most similar documents to the query. 
    The query is compared to the document embeddings in the vector store using cosine similarity. 
    The top k most similar documents are returned. If no results are found, the function returns "No results found".
    """

    embedder = OllamaEmbeddings(
        model="llama3.2:1b"
    )
    vector_store = FAISS.load_local(folder_path="./Knowledge/CM+Code_Conso/",
                                    embeddings=embedder,
                                    allow_dangerous_deserialization=True)
    
    raw = vector_store.similarity_search(query=q, k=k)
    return [r.page_content.replace("\n", " ").replace("  ", " ") for r in raw] if raw else "No results found"