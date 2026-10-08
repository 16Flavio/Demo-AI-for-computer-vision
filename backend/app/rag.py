import asyncio
import io
import json
import os
import secrets
import time

import httpx
import numpy as np
import onnxruntime as ort
from pypdf import PdfReader
from tokenizers import Tokenizer

MAX_PAGES = 20
MOTS_PAR_PASSAGE = 120
CHEVAUCHEMENT = 20
NB_PASSAGES = 4
DUREE_SESSION = 30 * 60  # secondes : le document est oublié ensuite
MAX_DOCUMENTS = 20       # documents gardés en mémoire en même temps
DELAI_GENERATION = 90    # secondes

LLM_URL = os.environ.get("LLM_URL", "http://127.0.0.1:8080")

DOSSIER_E5 = "../models/multilingual-e5-small/onnx"
session_e5 = ort.InferenceSession(f"{DOSSIER_E5}/model.onnx")
tokenizer = Tokenizer.from_file(f"{DOSSIER_E5}/tokenizer.json")
tokenizer.enable_truncation(512)
tokenizer.enable_padding()

# Les documents restent en mémoire uniquement : rien n'est écrit sur le disque
documents = {}
verrou_llm = asyncio.Lock()  # une seule génération à la fois, les autres attendent


# ---------- Embeddings ----------
def embeddings(textes, prefixe):
    """multilingual-e5 attend « query: » pour une question et « passage: » pour un passage."""
    vecteurs = []
    for i in range(0, len(textes), 16):
        lot = tokenizer.encode_batch([prefixe + t for t in textes[i:i + 16]])
        ids = np.array([e.ids for e in lot], dtype=np.int64)
        masque = np.array([e.attention_mask for e in lot], dtype=np.int64)
        sortie = session_e5.run(None, {"input_ids": ids, "attention_mask": masque,
                                       "token_type_ids": np.zeros_like(ids)})[0]
        moyenne = (sortie * masque[..., None]).sum(1) / masque.sum(1, keepdims=True)
        vecteurs.append(moyenne / np.linalg.norm(moyenne, axis=1, keepdims=True))
    return np.concatenate(vecteurs)


# ---------- Lecture et découpage du PDF ----------
def decouper(pages):
    """Découpe le texte en passages de quelques centaines de mots, en gardant le numéro de page."""
    passages = []
    for numero, texte in enumerate(pages, start=1):
        mots = texte.split()
        pas = MOTS_PAR_PASSAGE - CHEVAUCHEMENT
        for debut in range(0, max(len(mots) - CHEVAUCHEMENT, 1), pas):
            morceau = " ".join(mots[debut:debut + MOTS_PAR_PASSAGE])
            if morceau:
                passages.append({"page": numero, "texte": morceau})
    return passages


def nettoyer_sessions():
    maintenant = time.time()
    for cle in [c for c, d in documents.items() if d["expire"] < maintenant]:
        del documents[cle]
    while len(documents) >= MAX_DOCUMENTS:  # on oublie le plus ancien
        del documents[min(documents, key=lambda c: documents[c]["expire"])]


def ajouter_document(contenu, nom):
    lecteur = PdfReader(io.BytesIO(contenu))
    if len(lecteur.pages) > MAX_PAGES:
        raise ValueError(f"Le PDF a {len(lecteur.pages)} pages (maximum {MAX_PAGES}).")
    passages = decouper([page.extract_text() or "" for page in lecteur.pages])
    if not passages:
        raise ValueError("Aucun texte trouvé dans ce PDF (document scanné ?).")

    vecteurs = embeddings([p["texte"] for p in passages], "passage: ")

    nettoyer_sessions()
    identifiant = secrets.token_urlsafe(16)
    documents[identifiant] = {"nom": nom, "passages": passages, "vecteurs": vecteurs,
                              "expire": time.time() + DUREE_SESSION}
    return {"id": identifiant, "nom": nom, "pages": len(lecteur.pages), "passages": len(passages)}


def supprimer_document(identifiant):
    documents.pop(identifiant, None)


# ---------- Recherche ----------
def rechercher(identifiant, question):
    nettoyer_sessions()
    document = documents.get(identifiant)
    if document is None:
        raise KeyError("Document introuvable ou expiré : renvoyez le PDF.")
    document["expire"] = time.time() + DUREE_SESSION

    q = embeddings([question], "query: ")[0]
    similarites = document["vecteurs"] @ q  # vecteurs normalisés : produit scalaire = cosinus
    meilleurs = np.argsort(-similarites)[:NB_PASSAGES]
    return [{**document["passages"][i], "score": round(float(similarites[i]), 3)} for i in meilleurs]


# ---------- Réponse du LLM ----------
CONSIGNE = (
    "Tu réponds à des questions sur un document. Utilise UNIQUEMENT les passages fournis. "
    "Après chaque information, cite sa source entre crochets, par exemple [1] ou [2][3]. "
    "Si les passages ne contiennent pas la réponse, dis-le simplement. "
    "Réponds dans la langue de la question, de façon concise."
)


def construire_messages(question, passages):
    contexte = "\n\n".join(f"[{i}] (page {p['page']}) {p['texte']}" for i, p in enumerate(passages, start=1))
    return [
        {"role": "system", "content": CONSIGNE},
        {"role": "user", "content": f"Passages :\n{contexte}\n\nQuestion : {question}"},
    ]


async def repondre(identifiant, question):
    """Générateur de lignes JSON : passages trouvés, puis la réponse morceau par morceau."""
    passages = await asyncio.to_thread(rechercher, identifiant, question)
    yield json.dumps({"passages": passages}) + "\n"

    if verrou_llm.locked():
        yield json.dumps({"attente": True}) + "\n"

    async with verrou_llm:
        requete = {"messages": construire_messages(question, passages), "stream": True,
                   "max_tokens": 600, "temperature": 0.2}
        debut = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(DELAI_GENERATION, connect=5)) as client:
                async with client.stream("POST", f"{LLM_URL}/v1/chat/completions", json=requete) as reponse:
                    reponse.raise_for_status()
                    async for ligne in reponse.aiter_lines():
                        if not ligne.startswith("data: ") or ligne == "data: [DONE]":
                            continue
                        morceau = json.loads(ligne[6:])["choices"][0]["delta"].get("content")
                        if morceau:
                            yield json.dumps({"texte": morceau}) + "\n"
                        if time.monotonic() - debut > DELAI_GENERATION:
                            yield json.dumps({"erreur": "Délai de génération dépassé."}) + "\n"
                            return
        except httpx.HTTPError:
            yield json.dumps({"erreur": "Le modèle de langage est indisponible pour le moment."}) + "\n"
            return
    yield json.dumps({"fin": True}) + "\n"
