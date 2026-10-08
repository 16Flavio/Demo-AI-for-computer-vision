"""Télécharge tous les modèles des démos.

- models/            : modèles utilisés par le serveur (hors Git)
- frontend/modeles/  : modèle de vision téléchargé par le navigateur (hors Git)
- frontend/videos/   : vidéo d'exemple pour la démo sans webcam (hors Git)
- frontend/vendor/   : ONNX Runtime Web, servi par le site lui-même pour le calcul sur plusieurs fils (hors Git)

Usage : python scripts/download_models.py [--sans-llm]
"""
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image

RACINE = Path(__file__).resolve().parent.parent
MODELS_DIR = RACINE / "models"
MODELES_WEB = RACINE / "frontend" / "modeles"
VIDEOS_WEB = RACINE / "frontend" / "videos"

# YOLOX-Nano (Megvii, licence Apache 2.0)
YOLOX_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx"
# ONNX Runtime Web (licence MIT). Servi depuis le site (même origine) : le WebAssembly sur plusieurs
# fils d'exécution exige une page isolée (en-têtes COOP/COEP), qui bloque les workers d'un autre domaine.
ORT_WEB_VERSION = "1.30.0"
ORT_WEB_FICHIERS = ["ort.wasm.min.js", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"]
VENDOR_WEB = RACINE / "frontend" / "vendor" / f"onnxruntime-web-{ORT_WEB_VERSION}"
# Vidéo d'exemple : « Diagonal crosswalk Yonge Dundas », Wikimedia Commons, domaine public (CC0)
VIDEO_URL = ("https://upload.wikimedia.org/wikipedia/commons/transcoded/d/d2/DiagonalCrosswalkYongeDundas.webm/"
             "DiagonalCrosswalkYongeDundas.webm.360p.vp9.webm")


def telecharger(url, destination):
    if destination.exists():
        print(f"Déjà présent : {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    requete = urllib.request.Request(url, headers={"User-Agent": "labo-demo/1.0"})
    with urllib.request.urlopen(requete) as reponse, open(destination, "wb") as fichier:
        fichier.write(reponse.read())
    print(f"Téléchargé : {destination}")


def images_calibration():
    """Images 416x416 (BGR, 0-255, fond gris comme dans le navigateur) pour calibrer la quantification int8.

    Des images de la vidéo d'exemple (une toutes les 2 secondes, via ffmpeg s'il est installé),
    complétées par des recadrages de la photo d'exemple.
    """
    images = []
    if shutil.which("ffmpeg"):
        with tempfile.TemporaryDirectory() as dossier:
            subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", "0.5", "-i", str(VIDEOS_WEB / "exemple.webm"),
                            "-vf", "fps=1/2", f"{dossier}/%03d.png"], check=True)
            images = [Image.open(f).convert("RGB") for f in sorted(Path(dossier).glob("*.png"))]
    photo = Image.open(RACINE / "frontend" / "DemoImage.png").convert("RGB")
    rng = np.random.default_rng(0)
    for _ in range(16):
        l, h = photo.size
        taille = int(min(l, h) * rng.uniform(0.5, 1.0))
        x, y = rng.integers(0, l - taille + 1), rng.integers(0, h - taille + 1)
        images.append(photo.crop((x, y, x + taille, y + taille)))

    for image in images:
        echelle = min(416 / image.width, 416 / image.height)
        fond = Image.new("RGB", (416, 416), (114, 114, 114))
        fond.paste(image.resize((int(image.width * echelle), int(image.height * echelle))), (0, 0))
        tableau = np.array(fond, dtype=np.float32)[:, :, ::-1]  # RGB -> BGR
        yield np.ascontiguousarray(tableau.transpose(2, 0, 1)[np.newaxis])


def noeuds_en_float(modele):
    """Noeuds laissés en virgule flottante : ceux qui perdent le plus de précision en int8.

    - le début du réseau, jusqu'à la première connexion résiduelle (Add) : il reçoit l'image brute (0 à 255).
      Le quantifier fait chuter le score « objet » : seuls 56 % des objets du modèle normal étaient retrouvés.
    - la fin, après les dernières convolutions : elle concatène positions (dizaines) et probabilités (0 à 1),
      qu'une seule échelle int8 ne peut pas représenter correctement.
    """
    noeuds = modele.graph.node
    premier_add = next(i for i, n in enumerate(noeuds) if n.op_type == "Add")
    garder = {n.name for n in noeuds[:premier_add + 1]}

    producteur = {sortie: n for n in noeuds for sortie in n.output}
    pile = [modele.graph.output[0].name]
    while pile:
        n = producteur.get(pile.pop())
        if n is None or n.name in garder or n.op_type == "Conv":
            continue
        garder.add(n.name)
        pile += list(n.input)
    return sorted(garder)


def quantifier_yolox(source, destination):
    if destination.exists():
        print(f"Déjà présent : {destination}")
        return
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Lecteur(CalibrationDataReader):
        def __init__(self):
            self.images = iter(images_calibration())

        def get_next(self):
            image = next(self.images, None)
            return None if image is None else {"images": image}

    import onnx
    from onnx import version_converter

    # Le modèle d'origine est en opset 11 ; la quantification par canal demande l'opset 13
    pretraite = destination.with_suffix(".pre.onnx")
    onnx.save(version_converter.convert_version(onnx.load(source), 13), pretraite)
    quant_pre_process(str(pretraite), str(pretraite), skip_symbolic_shape=True)
    quantize_static(str(pretraite), str(destination), Lecteur(),
                    quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    per_channel=True, nodes_to_exclude=noeuds_en_float(onnx.load(pretraite)))
    pretraite.unlink()
    print(f"Quantifié : {destination}")


def main():
    sans_llm = "--sans-llm" in sys.argv

    # Démo profondeur
    hf_hub_download(
        repo_id="onnx-community/depth-anything-v2-small",
        filename="onnx/model.onnx",
        local_dir=MODELS_DIR / "depth-anything-v2-small",
    )

    # Démo 2 : détection d'objets dans le navigateur
    telecharger(YOLOX_URL, MODELES_WEB / "yolox_nano.onnx")
    telecharger(VIDEO_URL, VIDEOS_WEB / "exemple.webm")
    for fichier in ORT_WEB_FICHIERS:
        telecharger(f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{ORT_WEB_VERSION}/dist/{fichier}", VENDOR_WEB / fichier)
    quantifier_yolox(MODELES_WEB / "yolox_nano.onnx", MODELES_WEB / "yolox_nano_int8.onnx")

    # Démo 3 : embeddings multilingues (licence MIT)
    for fichier in ["onnx/model.onnx", "onnx/tokenizer.json"]:
        hf_hub_download(
            repo_id="intfloat/multilingual-e5-small",
            filename=fichier,
            local_dir=MODELS_DIR / "multilingual-e5-small",
        )

    # Démo 3 : LLM Qwen3-4B-Instruct-2507, quantifié en 4 bits (licence Apache 2.0, ~2,5 Go)
    if not sans_llm:
        hf_hub_download(
            repo_id="unsloth/Qwen3-4B-Instruct-2507-GGUF",
            filename="Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
            local_dir=MODELS_DIR / "llm",
        )

    print("Tous les modèles sont prêts.")


if __name__ == "__main__":
    main()
