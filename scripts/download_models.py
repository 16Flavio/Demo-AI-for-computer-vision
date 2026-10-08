from pathlib import Path
from huggingface_hub import hf_hub_download

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

def main():

    path = hf_hub_download(
        repo_id = "onnx-community/depth-anything-v2-small",
        filename = "onnx/model.onnx",
        local_dir = MODELS_DIR / "depth-anything-v2-small",
    )

    print(f"Model downloaded : {path}")

if __name__ == "__main__":
    main()