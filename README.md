# Demo AI

Demo of computer vision and optimisation.

## Start in local
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python ../scripts/download_models.py
uvicorn app.main:app --reload