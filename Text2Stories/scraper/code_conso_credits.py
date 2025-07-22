import requests
from dotenv import load_dotenv, set_key
import os, sys, subprocess
import time
import random
import pandas as pd

root = subprocess.run(["git", "rev-parse", "--show-toplevel"],
capture_output=True,
text=True,
encoding="utf-8").stdout.strip()

sys.path.append(root)

load_dotenv()

CREATE_TOKEN = True

client_id = os.getenv("PISTE_ID")
client_secret = os.getenv("PISTE_SECRET")
book_index = 2
title_index = 0

if CREATE_TOKEN:
    url_auth = "https://oauth.piste.gouv.fr/api/oauth/token"
    data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret
    }

    response = requests.post(url_auth, data=data)
    token_data = response.json()
    access_token = token_data.get("access_token")
    os.environ["PISTE_KEY"] = access_token
    set_key('../.env', "PISTE_KEY", access_token)
else: 
    access_token = os.getenv("PISTE_KEY")

url_api = "https://api.piste.gouv.fr/dila/legifrance/lf-engine-app/consult/code"
headers = {
    "Authorization": f"Bearer {access_token}",
    "Content-Type": "application/json",
    "accept": "application/json",
}

payload = {
  "textId": "LEGITEXT000006069565",
  "sctCid": "LEGISCTA000032226222",
  "date": "2025-03-20"
}

response = requests.post(url_api, json=payload, headers=headers)
code_conso = response.json()


articles_id = []
def get_articles_id(code, articles_id: list = []):
    if code["sections"] not in [None, []]:
        for section in code["sections"]:
            get_articles_id(section, articles_id=articles_id)
    else:
        for article in code["articles"]:
            articles_id.append(article["id"])

get_articles_id(code_conso["sections"][0]["sections"][book_index]["sections"][title_index], articles_id)


def get_article(ids: list):
    url_api = "https://api.piste.gouv.fr/dila/legifrance/lf-engine-app/consult/getArticle"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "accept": "application/json"
    }

    articles = []
    errors = 0
    for id in ids:
        payload = {
            "id": id
        }
        try: 
            response = requests.post(url_api, json=payload, headers=headers)
            articles.append(response.json()["article"]["texte"])
        except:
            print("An error occured")
            errors += 1
        
        time.sleep(random.uniform(0.1, 0.3))
        
    return articles, errors

texts, errors = get_article(articles_id)

loan_articles = pd.DataFrame(index=articles_id, data=texts, columns=["Article"])
loan_articles.to_csv(root+"/Data/CodeConso/loan_articles.csv")