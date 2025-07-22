import subprocess
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from langchain_ollama import OllamaLLM

root = subprocess.run(["git", "rev-parse", "--show-toplevel"], 
capture_output=True, 
text=True,
encoding="utf-8").stdout.strip()

EXTRACT = "Restructure l'information dans le texte suivant : \n\n {content}"
model = OllamaLLM(model="qwen2.5:3b", device="cuda:0")
parser = StrOutputParser()

raw_path = root + "/Data/Raw"
processed_path = root + "/Data/Processed"
os.makedirs(processed_path, exist_ok=True)

for folder in os.listdir(raw_path):
    save_path = processed_path + "/" + folder
    for file in os.listdir(raw_path + "/" + folder):
        with open(raw_path + "/" + folder + "/" + file, "r", encoding="utf-8") as f:
            data = f.read()
        prompt = ChatPromptTemplate([
            ("system", EXTRACT)
        ])
        #DEBUG
        #print(prompt.format(**{"theme": folder, "content": data}))
        chain = prompt | model | parser
        output = chain.invoke(input = {"theme": folder, "content": data})
        
        os.makedirs(save_path, exist_ok=True)
        with open(save_path + "/" + file, "w", encoding="utf-8") as f:
            f.write(output)