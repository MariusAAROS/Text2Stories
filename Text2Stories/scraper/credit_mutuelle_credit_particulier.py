import os
import time
import random
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Dictionnaire des types de crédits et leurs URLs correspondantes
credits = {
    "credit_immobilier": "https://www.creditmutuel.fr/fr/particuliers/credits/credit-immobilier.html",
    "pret_travaux": "https://www.creditmutuel.fr/fr/particuliers/credits/credit-travaux.html",
    "credit_conso": "https://www.creditmutuel.fr/fr/particuliers/credits/credit-conso.html",
    "credit_voiture": "https://www.creditmutuel.fr/fr/particuliers/credits/credits-voiture.html",
    "credit_velo": "https://www.creditmutuel.fr/fr/particuliers/credits/credit-velo.html",
    "pret_etudiant": "https://www.creditmutuel.fr/fr/particuliers/credits/pret-etudiant.html",
}

# Fonction pour extraire et sauvegarder le texte d'une page web
def extraire_et_sauvegarder(url, nom_dossier, nom_fichier):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")

        # Suppression des scripts et styles
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()

        texte = soup.get_text(separator="\n", strip=True)

        # Nettoyage du nom du fichier pour éviter les erreurs
        nom_fichier = nom_fichier.replace(" ", "_").replace("/", "_").replace(":", "_").replace("\n", "_")

        os.makedirs(f"../Data/Raw/{nom_dossier}", exist_ok=True)  # Crée le dossier s'il n'existe pas
        with open(f"../Data/Raw/{nom_dossier}/{nom_fichier}.txt", "w", encoding="utf-8") as fichier:
            fichier.write(texte)
        print(f"✅ Contenu de {url} sauvegardé dans {nom_dossier}/{nom_fichier}.txt")
    else:
        print(f"❌ Échec de la récupération de {url} (statut {response.status_code})")

options = webdriver.ChromeOptions()
options.add_argument("--headless")  # Mode headless (optionnel)
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
selectors = {
    "credit_immobilier": "#-AF-Hub-Conv-Part-Pret-immobilier > section.page__element.tilesessentiels > div > div.tilesessentiels__content-container > ul",
    "pret_travaux": "#-AF-Hub-Conv-Part-Credit-travaux > section.page__element.tilesessentiels > div > div.tilesessentiels__content-container > ul",
    "credit_conso": "#-AF-Hub-Conv-Part-Credits-a-la-consommation > section.page__element.tilesessentiels > div > div.tilesessentiels__content-container > ul",
    "credit_voiture": "#-AF-Hub-Conv-Part-Credits-voiture > section.page__element.tilesessentiels > div > div.tilesessentiels__content-container > ul",
    "credit_velo": None,
    "pret_etudiant": None,
}

for (nom, url), (_, section_selector) in zip(credits.items(), selectors.items()):
    extraire_et_sauvegarder(url, nom, nom)
    driver.get(url)
    wait = WebDriverWait(driver, 10)

    # Sélecteur de la section principale des produits de prêt
    #section_selector = "#-AF-Hub-Conv-Part-Pret-immobilier > section.page__element.tilesessentiels > div > div.tilesessentiels__content-container > ul"
    if section_selector is None:
        print(f"⚠️ Sélecteur non défini pour {nom}")
        continue
    try:
        section = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, section_selector)))
        product_selector = "div.tilesessentiels__content-container > ul > li"
        products = section.find_elements(By.CSS_SELECTOR, product_selector)
    except Exception as e:
        print(f"❌ Erreur lors de la récupération des produits: {e}")
        continue

    for index, product in enumerate(products):
        try:
            # 🛠️ Solution pour "stale element reference"
            section = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, section_selector)))
            products = section.find_elements(By.CSS_SELECTOR, product_selector)
            product = products[index]

            driver.execute_script("arguments[0].scrollIntoView();", product)
            time.sleep(1)

            p_text = product.text.strip()

            # Nettoyage du texte du produit pour un nom de fichier valide
            p_text_clean = p_text.split("\n")[0].replace(" ", "_").replace("/", "_").replace(":", "_").replace("\n", "_")

            print(f"🔍 Processing: {p_text_clean}")

            try:
                button = product.find_element(By.CSS_SELECTOR, "div.tile__produit__actions > a")
            except:
                print("⚠️ Bouton non trouvé pour ce produit")
                continue

            driver.execute_script("arguments[0].scrollIntoView();", button)
            wait.until(EC.element_to_be_clickable(button))

            try:
                button.click()
            except:
                driver.execute_script("arguments[0].click();", button)

            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "body")))

            extraire_et_sauvegarder(driver.current_url, nom, f"{p_text_clean}")

            driver.back()
            #wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, section_selector)))

            time.sleep(1 + random.random() * 2)

        except Exception as e:
            print(f"❌ Erreur lors du traitement de {p_text_clean}: {e}")
            continue

driver.quit()