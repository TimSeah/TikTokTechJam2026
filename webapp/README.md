# Real or Fake: Human vs AI

A standalone React + FastAPI challenge app. Each round times the detector before revealing the
image, then measures the player's answer from the moment the image finishes loading. After ten
rounds it compares both accuracy and average response time.

## Configure

Copy `.env.example` to `.env`. The deployable detector is selected with
`MODEL_ARTIFACT_PATH`; point it at a compatible fine-tuned `joblib` bundle when that model is
ready. The bundle supplies its matching OpenCLIP model name, checkpoint, feature dimensions, and
threshold. Only load model artifacts that you trust because `joblib` files can execute code while
loading.

`CHALLENGE_DATASET_PATH` must contain images below directories named `REAL` and `FAKE`.

## Develop

From the repository root, run the API with the existing Python environment:

```powershell
.\.venv-amd\Scripts\python.exe -m pip install -r webapp\requirements.txt
.\.venv-amd\Scripts\python.exe -m uvicorn webapp.backend.main:app --reload
```

In a second terminal, run the frontend:

```powershell
Set-Location webapp\frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Production

Build the frontend, then start FastAPI. The API serves the compiled frontend from the same origin.

```powershell
Set-Location webapp\frontend
npm ci
npm run build
Set-Location ..\..
.\.venv-amd\Scripts\python.exe -m uvicorn webapp.backend.main:app --host 0.0.0.0 --port 8000
```
