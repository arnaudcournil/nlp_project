import pandas as pd
from spellchecker import SpellChecker
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import FrenchStemmer
import nltk
from unidecode import unidecode
import ast
from transformers import pipeline
from rank_bm25 import BM25Okapi
import streamlit as st
import pickle
import numpy as np

nltk.download('stopwords')
nltk.download('punkt')


try:
    useless_words = pd.read_csv("most_frequent_words_mixed.csv", header=None)[0].tolist()[:100]
except:
    useless_words = []

train = pd.read_csv("reviews_train.csv")
train["review_text"] = train["review_text"].apply(lambda x: ast.literal_eval(x))
# Liste d'avis
documents = train["review_text"].tolist()

# Listes de scores
ratings = train["review_rating"].tolist()

# Créer un modèle BM25
bm25 = BM25Okapi(documents)

STEMMER = FrenchStemmer()
spell = SpellChecker(language='fr')
pipe = pipeline("text-classification", model="tblard/tf-allocine")

BOTH_MODEL = pickle.load(open("both_model.pkl", "rb"))
PIPE_MODEL = pickle.load(open("pipe_model.pkl", "rb"))
BM25_MODEL = pickle.load(open("bm25_model.pkl", "rb"))

def preprocess_text(text):
    # Suppression des accents
    text = unidecode(text)
    # Suppression du code HTML
    text = re.sub(re.compile("<.*?>"), "", text)
    text = re.sub(r'[^a-zA-Z0-9/s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # Suppresssion des nombres
    text = re.sub(r'[0-9]+', ' ', text)
    # Supprimer les lignes vides
    text = text.split('\n')
    text = [line.strip() for line in text if len(line) > 0]
    text = ' '.join(text)
    # Supprimer les liens
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    # Lemmatiser les mots
    tokens = word_tokenize(text.lower(), language='french')
    return tokens

n_grams = lambda tokens, n: [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def text2Token(text, spelling = True, stem = True, len_word_min = 2, spell = spell, useless_words = useless_words):
    stopword = stopwords.words('french')
    word_tokens = preprocess_text(text)
    word_tokens = [word for word in word_tokens if word not in stopword and word not in useless_words and len(word) > len_word_min]
    if spelling:
        word_tokens = [spell.correction(word) for word in word_tokens]
        word_tokens = [word for word in word_tokens if word != None]
    if stem:
        word_tokens = [STEMMER.stem(token) for token in word_tokens]
    word_tokens_with_n_grams = word_tokens + n_grams(word_tokens, 2) + n_grams(word_tokens, 3)
    return word_tokens_with_n_grams

def getMostFrequentWords(documents, top=10):
    # Compter les mots
    word_count = {}
    for doc in documents:
        for word in doc:
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1

    # Trier les mots par fréquence décroissante
    word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

    if top == float('inf'):
        return word_count
    
    return word_count[:top]

def getTopDocs(bm25, query, documents, ratings, top=5):
    # Calculer les scores de similarité
    scores = bm25.get_scores(query)

    # Associer chaque avis à son score
    doc_scores = list(zip(documents, scores, ratings))

    # Trier les avis par score décroissant
    return sorted(doc_scores, key=lambda x: x[1], reverse=True)[:top]

def separer_phrase(phrase):
    # On ajoute des points aux sauts de ligne
    phrase = phrase.replace('\n', '.')

    # Divise d'abord la phrase en utilisant les points, points d'interrogation, points d'exclamation.
    pattern = r'(?<=[.!?])(?=\s|[A-Z"\'(])'
    groupes = re.split(pattern, phrase)

    groupes_fins = []
    for groupe in groupes:
        # Combinaison des motifs de virgule et "et" en une seule expression régulière
        # Sépare sur les virgules (en évitant les nombres décimaux), sur les ; et : et sur les conjonctions de coordinations (avec un contexte spécifique)
        pattern_combined = r'(?<=.{20},)\s(?!\d)|[;:]|\b(mais|ou|et|donc|or|ni|car)\b(?=.{15,})'
        sous_groupes = re.split(pattern_combined, groupe)
        groupes_fins.extend(sous_groupes)

    return [groupe.strip() for groupe in groupes_fins if groupe is not None and groupe.strip() and groupe.strip() not in ['.', ',', 'mais', 'ou', 'et', 'donc', 'or', 'ni', 'car']]

def estimate_score(top_docs, origin_query = None, use_bm25 = True, FIABILITY_THRESHOLD = 0.6):
    # Calculer la note moyenne des avis
    if use_bm25 and origin_query is None:
        return BM25_MODEL.predict(np.array([top_docs[i][2] for i in range(len(top_docs))] + [top_docs[i][1] for i in range(len(top_docs))]).reshape(1, -1))
    elif not use_bm25:
        pipe_res = pipe(origin_query)[0]
        return PIPE_MODEL.predict(np.array([pipe_res["score"], 1 if pipe_res["label"] == "POSITIVE" else 0]).reshape(1, -1))
    else:
            pipe_res = pipe(origin_query)[0]
            pipe_score = pipe_res["score"]
            pipe_label = 1 if pipe_res["label"] == "POSITIVE" else 0
            bm25_ratings = [top_docs[i][2] for i in range(len(top_docs))]
            bm25_scores = [top_docs[i][1] for i in range(len(top_docs))]
            return BOTH_MODEL.predict(np.array([pipe_score, pipe_label] + bm25_ratings + bm25_scores).reshape(1, -1))
    
def getRevelantSentences(origin_query, most_freq, documents, ratings, top=5, use_bm25 = True, use_pipe = True):

    # Appel de la fonction
    groupes = separer_phrase(origin_query)

    # Obtenir les scores de chaque groupe
    scores = []
    for groupe in groupes:
        if use_pipe and use_bm25:
            scores.append(estimate_score(getTopDocs(bm25, text2Token(groupe), documents, ratings), origin_query))
        elif use_pipe:
            scores.append(estimate_score(None, origin_query, use_bm25=False))
        elif use_bm25:
            scores.append(estimate_score(getTopDocs(bm25, text2Token(groupe), documents, ratings)))
            
    pos_list = []
    neg_list = []
    for groupe, score in zip(groupes, scores):
        group_tokens = text2Token(groupe)
        sumFreq = sum([freq for word, freq in most_freq if word in group_tokens])
        if score >= 3.5:
            pos_list.append((groupe, sumFreq))
        elif score <= 1.5:
            neg_list.append((groupe, sumFreq))

    pos_list = [sentence[0] for sentence in sorted(pos_list, key=lambda x: x[1], reverse=True)[:top]]
    neg_list = [sentence[0] for sentence in sorted(neg_list, key=lambda x: x[1], reverse=True)[:top]]

    return pos_list, neg_list

def main(origin_query, bm25=bm25, documents=documents, ratings=ratings, spell=spell, use_bm25 = True, use_pipe = True):
    query = text2Token(origin_query)
    top_docs = []
    most_freq = []
    if use_bm25:
        top_docs = getTopDocs(bm25, query, documents, ratings, top=5)
        most_freq = getMostFrequentWords([doc[0] for doc in top_docs], top=50)
    pos_list, neg_list = getRevelantSentences(origin_query, most_freq, documents, ratings, top=5, use_bm25 = True, use_pipe = True)
    if not use_pipe:
        origin_query = None
    return estimate_score(top_docs, origin_query, use_bm25=use_bm25), pos_list, neg_list

"""
Streamlit
"""

def prediction_1(origin_query):
    try:
        score, pos_list, neg_list = main(origin_query, use_pipe=False)
    except:
        score, pos_list, neg_list = 0, [], []
    return {
        "nombre d étoile sur 5": score,
        "liste phrases positives": pos_list,
        "liste phrases négatives": neg_list
    }

def prediction_2(origin_query):
    try:
        score, pos_list, neg_list = main(origin_query, use_bm25=False)
    except:
        score, pos_list, neg_list = 0, [], []
    return {
        "nombre d étoile sur 5": score,
        "liste phrases positives": pos_list,
        "liste phrases négatives": neg_list
    }

def prediction_3(origin_query):
    try:
        score, pos_list, neg_list = main(origin_query)
    except:
        score, pos_list, neg_list = 0, [], []
    return {
        "nombre d étoile sur 5": score,
        "liste phrases positives": pos_list,
        "liste phrases négatives": neg_list
    }


def afficher_resultats(resultats):
    if resultats["nombre d étoile sur 5"] == 0:
        st.write("La prédiction n'a pas pu être effectuée.")
        return None
    st.subheader("Résultats de la prédiction :")
    st.write(f"Nombre d'étoiles sur 5 : {'🌟' * round(resultats['nombre d étoile sur 5'])}")
    st.subheader("Liste de phrases positives :")
    for phrase in resultats["liste phrases positives"]:
        st.write(f"👍 {phrase}")
    st.subheader("Liste de phrases négatives :")
    for phrase in resultats["liste phrases négatives"]:
        st.write(f"👎 {phrase}")

def run():
    st.title("Analyse de sentiments pour des avis de concessionnaires automobiles")

    avis_utilisateur = st.text_area("Entrez votre avis ici :")

    if st.button("Prédiction BM 25"):
        resultats_1 = prediction_1(avis_utilisateur)
        afficher_resultats(resultats_1)

    if st.button("Prédiction Transformers"):
        resultats_2 = prediction_2(avis_utilisateur)
        afficher_resultats(resultats_2)

    if st.button("Prédiction BM 25 + Transformers"):
        resultats_3 = prediction_3(avis_utilisateur)
        afficher_resultats(resultats_3)

run()