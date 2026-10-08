import io
from fastapi import FastAPI, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from app.depth import compute

app = FastAPI()

@app.post("/api/depth")
async def depth(fichier: UploadFile):
    contenu = await fichier.read()
    image = Image.open(io.BytesIO(contenu)).convert("RGB")

    depth_map = compute(image)

    tampon = io.BytesIO()
    depth_map.save(tampon, format="PNG")
    return Response(tampon.getvalue(), media_type="image/png")

app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")