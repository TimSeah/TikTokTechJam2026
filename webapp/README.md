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

## Deploy the backend to Modal

Modal runs the FastAPI API on a T4 GPU. The deployment uses one container because active round IDs
are held in process memory. The container scales to zero after ten idle minutes. Cloudflare Pages
serves the production frontend separately.

Install and authenticate the Modal CLI from the repository root:

```powershell
uv pip install --python .\.venv-amd\Scripts\python.exe -r webapp\requirements-modal.txt
.\.venv-amd\Scripts\modal.exe setup
```

Create a named volume and upload the trusted classifier bundle and labeled challenge images. This
is a one-time operation unless either asset changes.

```powershell
.\.venv-amd\Scripts\modal.exe volume create human-vs-ai-assets
.\.venv-amd\Scripts\modal.exe volume put human-vs-ai-assets outputs\model.joblib /model.joblib
.\.venv-amd\Scripts\modal.exe volume put human-vs-ai-assets data\downloads\test /challenge
```

Deploy the API. Modal bakes the Python dependencies and current OpenCLIP checkpoint into its image;
the classifier and challenge images stay on the read-only volume.

```powershell
.\.venv-amd\Scripts\modal.exe deploy webapp\modal_app.py --strategy recreate
```

The backend is available at `https://timseah--human-vs-ai-play.modal.run`. To replace the
classifier later, run:

```powershell
.\.venv-amd\Scripts\modal.exe volume put --force human-vs-ai-assets outputs\model.joblib /model.joblib
.\.venv-amd\Scripts\modal.exe deploy --strategy recreate webapp\modal_app.py
```

If a new artifact uses a different OpenCLIP model or pretrained checkpoint, update
`OPENCLIP_MODEL_NAME` and `OPENCLIP_PRETRAINED` in `modal_app.py` before redeploying.

## Deploy the frontend to Cloudflare Pages

The production Vite build reads `VITE_API_BASE_URL` from `.env.production` and calls the Modal API
directly. Static HTML, JavaScript, CSS, and fonts are served by Cloudflare; model inference and
challenge images remain on Modal.

For the first deployment on a new machine, authenticate Wrangler. The `human-vs-ai` Pages project
already exists, so do not recreate it.

```powershell
npx --yes wrangler@4.127.1 login
```

Build and deploy the production frontend from the repository root:

```powershell
Set-Location webapp\frontend
npm ci
npm run build
Set-Location ..\..
npx --yes wrangler@4.127.1 pages deploy webapp\frontend\dist --project-name human-vs-ai --branch main
```

The stable frontend URL is `https://human-vs-ai-ce2.pages.dev`. Cloudflare may also print an
immutable deployment-specific URL after each upload.

The Modal frontend is not a separate process that can be stopped independently. This deployment
excludes frontend files from the Modal image, and the Modal root returns 404. Actual site visits
still call `/api/status`, which starts the GPU backend when it has scaled to zero. After ten idle
minutes, `min_containers=0` allows active Modal compute to return to zero.

## GitHub CI/CD

The workflows in `.github/workflows` validate matching pull requests and deploy matching changes
after they reach `main`. Both workflows can also be run manually from the Actions tab.

Add these encrypted repository secrets under **Settings > Secrets and variables > Actions**:

| Secret | Value |
| --- | --- |
| `CLOUDFLARE_ACCOUNT_ID` | The Cloudflare account ID that owns the `human-vs-ai` Pages project. |
| `CLOUDFLARE_API_TOKEN` | A Cloudflare API token scoped to that account with Workers/Pages edit access. |
| `MODAL_TOKEN_ID` | The ID of a dedicated Modal API token. |
| `MODAL_TOKEN_SECRET` | The secret for the same Modal API token. |

Create the Cloudflare token from **Cloudflare Dashboard > Manage Account > API Tokens**. Create the
Modal token from **Modal Dashboard > Settings > API Tokens**. Enter token values directly in GitHub;
never add them to this repository or a local environment file.

`cloudflare-pages.yml` runs frontend lint and build checks, deploys `webapp/frontend/dist` to the
existing `human-vs-ai` Pages project, and verifies the production URL. `modal-backend.yml` runs the
backend tests, uploads the tracked `outputs/model.joblib` to the existing Volume, recreates the
Modal deployment, and verifies `/api/status`. Challenge images remain in the persistent Volume and
are not uploaded by CI.
