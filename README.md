# Labo de démos IA

Site : https://demo.flaviodrogo.be

Trois démos interactives, chacune illustrant une compétence :

| Démo | Page | Technologies | Où tourne le calcul |
|---|---|---|---|
| Tournées de livraison | `tournees.html` | OR-Tools, Leaflet, OpenStreetMap | Serveur |
| Analyse vidéo en direct | `webcam.html` | YOLOX-Nano, ONNX Runtime Web | Navigateur |
| Questions sur un PDF (RAG) | `rag.html` | llama.cpp, Qwen3-4B, multilingual-e5-small, pypdf | Serveur |
| Bonus : photo → 3D | `profondeur.html` | Depth Anything V2, three.js | Serveur |

## Télécharger les modèles (une fois, en local et sur le serveur)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt -r scripts/requirements.txt
python scripts/download_models.py
```

Le script remplit `models/` (modèles serveur, dont le LLM de ~2,5 Go), ainsi que `frontend/modeles/`, `frontend/videos/`
et `frontend/vendor/` (modèle de vision, vidéo d'exemple et ONNX Runtime Web, servis au navigateur).
Aucun de ces dossiers n'est dans Git. La quantification int8 utilise `ffmpeg` s'il est installé
(calibration sur des images de la vidéo d'exemple).

## Lancer en local

### Option 1 : Docker, identique à la production (recommandé)

```bash
sudo docker compose up --build
```

Puis ouvrir http://localhost:2028

### Option 2 : sans Docker (pratique pour développer)

Dans un premier terminal, le LLM ([binaire llama.cpp](https://github.com/ggml-org/llama.cpp/releases), version CPU) :

```bash
llama-server -m models/llm/Qwen3-4B-Instruct-2507-Q4_K_M.gguf --port 8080 --ctx-size 4096 --parallel 1
```

Dans un second terminal, l'API et le site :

```bash
cd backend
source ../.venv/bin/activate
uvicorn app.main:app --reload
```

Puis ouvrir http://localhost:8000 (la webcam fonctionne sur `localhost` sans HTTPS).

## Déploiement

```bash
cd ~/DemoAI
git pull
python scripts/download_models.py   # seulement si de nouveaux modèles ont été ajoutés
sudo docker compose up -d --build
```

Les fichiers déjà présents ne sont pas retéléchargés. Pour régénérer le modèle int8 après une modification
de la quantification, supprimer d'abord `frontend/modeles/yolox_nano_int8.onnx`.

La page `webcam.html` est servie avec les en-têtes `Cross-Origin-Opener-Policy` et `Cross-Origin-Embedder-Policy`
(ajoutés par l'API) : ils activent le WebAssembly sur plusieurs fils d'exécution. nginx doit les laisser passer
(c'est le cas par défaut).

Le LLM tourne dans un second conteneur (`llm`) sans port publié : seule l'API y accède.
Pour les réponses en streaming, nginx ne doit pas mettre la réponse en tampon (l'API envoie
`X-Accel-Buffering: no`). Pensez à `client_max_body_size 12M;` dans nginx pour l'envoi des PDF et photos.

## Licences des modèles

- YOLOX-Nano : Apache 2.0
- Qwen3-4B-Instruct-2507 : Apache 2.0
- multilingual-e5-small : MIT
- Depth Anything V2 Small : Apache 2.0
- ONNX Runtime Web : MIT
- Vidéo d'exemple : Wikimedia Commons, domaine public (CC0)
- Photos d'exemple de la démo de profondeur (`frontend/exemples/`) : Wikimedia Commons, domaine public (CC0)
