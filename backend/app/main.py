import io
import mimetypes
import threading
import time
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from app import rag, tournees
from app.depth import compute

TAILLE_MAX_IMAGE = 10 * 1024 * 1024  # 10 Mo
TAILLE_MAX_PDF = 10 * 1024 * 1024    # 10 Mo
REQUETES_PAR_MINUTE = 30             # par visiteur, sur /api/

app = FastAPI(title="Labo de démos IA")

# Un seul calcul lourd à la fois (profondeur, tournées) : les autres attendent leur tour
verrou_calcul = threading.Lock()
ATTENTE_MAX = 15  # secondes


def calcul_exclusif(fonction, *args, **kwargs):
    if not verrou_calcul.acquire(timeout=ATTENTE_MAX):
        raise HTTPException(503, "Serveur occupé, réessayez dans quelques secondes.")
    try:
        return fonction(*args, **kwargs)
    finally:
        verrou_calcul.release()


# ---------- Limitation du nombre de requêtes par visiteur ----------
historique = defaultdict(deque)


def adresse_visiteur(request: Request):
    # Derrière Cloudflare + nginx, la vraie adresse est dans les en-têtes
    return (request.headers.get("cf-connecting-ip")
            or request.headers.get("x-real-ip")
            or (request.client.host if request.client else "inconnu"))


@app.middleware("http")
async def limiter_requetes(request: Request, call_next):
    if request.url.path.startswith("/api/"):
        maintenant = time.monotonic()
        requetes = historique[adresse_visiteur(request)]
        while requetes and requetes[0] < maintenant - 60:
            requetes.popleft()
        if len(requetes) >= REQUETES_PAR_MINUTE:
            return JSONResponse({"detail": "Trop de requêtes, patientez une minute."}, status_code=429)
        requetes.append(maintenant)
    return await call_next(request)


# ---------- Page webcam : isolation pour le WebAssembly sur plusieurs fils ----------
# SharedArrayBuffer (donc ONNX Runtime Web multi-fils) n'est disponible que si la page est isolée.
# Uniquement sur cette page : l'isolation bloquerait les tuiles OpenStreetMap de la démo des tournées.
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("application/wasm", ".wasm")


@app.middleware("http")
async def isoler_page_webcam(request: Request, call_next):
    reponse = await call_next(request)
    # Les fichiers d'ONNX Runtime aussi : ses workers (fils d'exécution) sont refusés par une page isolée
    # si leur script n'a pas lui-même ces en-têtes
    if request.url.path == "/webcam.html" or request.url.path.startswith("/vendor/"):
        reponse.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        # « credentialless » : autorise le style et la photo de flaviodrogo.be sans en-tête CORP
        reponse.headers["Cross-Origin-Embedder-Policy"] = "credentialless"
    return reponse


async def lire_fichier(fichier: UploadFile, taille_max):
    contenu = await fichier.read(taille_max + 1)
    if len(contenu) > taille_max:
        raise HTTPException(413, f"Fichier trop volumineux (maximum {taille_max // (1024 * 1024)} Mo).")
    return contenu


# ---------- Démo profondeur ----------
@app.post("/api/depth")
async def depth(fichier: UploadFile):
    contenu = await lire_fichier(fichier, TAILLE_MAX_IMAGE)
    try:
        image = Image.open(io.BytesIO(contenu)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Image illisible.")

    depth_map = await run_in_threadpool(calcul_exclusif, compute, image)

    tampon = io.BytesIO()
    depth_map.save(tampon, format="PNG")
    return Response(tampon.getvalue(), media_type="image/png")


# ---------- Démo 1 : tournées de livraison ----------
class DemandeTournees(BaseModel):
    depot: tuple[float, float]                                  # (latitude, longitude)
    livraisons: list[tuple[float, float]] = Field(min_length=1, max_length=tournees.MAX_POINTS - 1)
    vehicules: int = Field(1, ge=1, le=tournees.MAX_VEHICULES)
    capacite: int | None = Field(None, ge=1, le=tournees.MAX_POINTS)


@app.post("/api/tournees")
def api_tournees(demande: DemandeTournees):
    points = [demande.depot, *demande.livraisons]
    debut = time.perf_counter()
    try:
        resultat = calcul_exclusif(tournees.optimiser, points, demande.vehicules, demande.capacite)
    except ValueError as erreur:
        raise HTTPException(400, str(erreur))
    resultat["duree_s"] = round(time.perf_counter() - debut, 2)
    return resultat


# ---------- Démo 3 : mini-RAG ----------
@app.post("/api/rag/document")
async def rag_document(fichier: UploadFile):
    contenu = await lire_fichier(fichier, TAILLE_MAX_PDF)
    try:
        return await run_in_threadpool(calcul_exclusif, rag.ajouter_document, contenu, fichier.filename or "document.pdf")
    except ValueError as erreur:
        raise HTTPException(400, str(erreur))
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(400, "PDF illisible.")


@app.delete("/api/rag/document/{identifiant}")
def rag_supprimer(identifiant: str):
    rag.supprimer_document(identifiant)
    return {"supprime": True}


class Question(BaseModel):
    document: str
    question: str = Field(min_length=2, max_length=500)


@app.post("/api/rag/question")
async def rag_question(q: Question):
    if q.document not in rag.documents:
        raise HTTPException(404, "Document introuvable ou expiré : renvoyez le PDF.")
    return StreamingResponse(rag.repondre(q.document, q.question), media_type="application/x-ndjson",
                             headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})


app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")
