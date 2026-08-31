from pathlib import Path

import modal

APP_NAME = "human-vs-ai"
ASSET_VOLUME_NAME = "human-vs-ai-assets"
ASSET_MOUNT_PATH = "/assets"
REMOTE_WEBAPP_ROOT = "/root/webapp"
OPENCLIP_MODEL_NAME = "ViT-B-32-quickgelu"
OPENCLIP_PRETRAINED = "openai"

LOCAL_WEBAPP_ROOT = Path(__file__).resolve().parent


def _cache_openclip_weights() -> None:
    import open_clip

    open_clip.create_model_and_transforms(
        OPENCLIP_MODEL_NAME,
        pretrained=OPENCLIP_PRETRAINED,
    )


app = modal.App(APP_NAME)
assets = modal.Volume.from_name(ASSET_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(str(LOCAL_WEBAPP_ROOT / "requirements.txt"))
    .run_function(_cache_openclip_weights, memory=4096, timeout=900)
    .env(
        {
            "MODEL_ARTIFACT_PATH": f"{ASSET_MOUNT_PATH}/model.joblib",
            "MODEL_DEVICE": "cuda",
            "CHALLENGE_DATASET_PATH": f"{ASSET_MOUNT_PATH}/challenge",
            "ROUND_CACHE_SIZE": "500",
            "ALLOWED_ORIGINS": "*",
        }
    )
    .add_local_dir(
        LOCAL_WEBAPP_ROOT,
        REMOTE_WEBAPP_ROOT,
        copy=True,
        ignore=[
            ".env",
            "**/__pycache__",
            "tests",
            "frontend",
        ],
    )
    .add_local_dir(
        LOCAL_WEBAPP_ROOT.parent / "src",
        "/root/src",
        copy=True,
        ignore=["**/__pycache__"],
    )
    .workdir("/root")
)


@app.function(
    image=image,
    gpu="T4",
    cpu=2.0,
    memory=4096,
    volumes={
        ASSET_MOUNT_PATH: assets.with_mount_options(read_only=True),
    },
    min_containers=0,
    max_containers=1,
    scaledown_window=600,
    startup_timeout=600,
    timeout=600,
)
@modal.concurrent(max_inputs=32)
@modal.asgi_app(label="human-vs-ai-play")
def fastapi_app():
    from webapp.backend.main import app as web_app

    return web_app
