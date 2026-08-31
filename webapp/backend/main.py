from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import WEBAPP_ROOT, Settings
from .detector import DetectorService
from .game import ChallengeGame

settings = Settings.from_env()


@asynccontextmanager
async def lifespan(app: FastAPI):
    detector = DetectorService(settings)
    game = ChallengeGame(settings, detector)
    if game.images:
        detector.predict(game.images[0].path)
    app.state.game = game
    yield


app = FastAPI(
    title="Real or Fake: Human vs AI",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.allowed_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class GuessRequest(BaseModel):
    label: Literal["REAL", "FAKE"]
    elapsed_ms: float = Field(ge=0, le=360_000)


def get_game(request: Request) -> ChallengeGame:
    return request.app.state.game


@app.get("/favicon.ico", include_in_schema=False)
def get_favicon() -> Response:
    return Response(status_code=204)


@app.get("/api/status")
def get_status(game: ChallengeGame = Depends(get_game)) -> dict:
    return {
        "ready": bool(game.images),
        "model_name": game.detector.display_name,
        "device": game.detector.device.type,
        "dataset_size": len(game.images),
    }


@app.post("/api/rounds")
def create_round(game: ChallengeGame = Depends(get_game)) -> dict:
    try:
        round_id, _ = game.create_round()
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {
        "round_id": round_id,
        "image_url": f"/api/rounds/{round_id}/image",
    }


@app.get("/api/rounds/{round_id}/image")
def get_round_image(
    round_id: str, game: ChallengeGame = Depends(get_game)
) -> FileResponse:
    challenge_round = game.get_round(round_id)
    if challenge_round is None:
        raise HTTPException(
            status_code=404, detail="Round not found or already finished"
        )
    return FileResponse(challenge_round.image.path)


@app.post("/api/rounds/{round_id}/guess")
def submit_guess(
    round_id: str,
    guess: GuessRequest,
    game: ChallengeGame = Depends(get_game),
) -> dict:
    challenge_round = game.finish_round(round_id)
    if challenge_round is None:
        raise HTTPException(
            status_code=404, detail="Round not found or already finished"
        )

    prediction = challenge_round.prediction
    ground_truth = challenge_round.image.label
    return {
        "ground_truth": ground_truth,
        "human_label": guess.label,
        "human_correct": guess.label == ground_truth,
        "human_elapsed_ms": guess.elapsed_ms,
        "ai_label": prediction.label,
        "ai_correct": prediction.label == ground_truth,
        "ai_elapsed_ms": prediction.elapsed_ms,
        "fake_probability": prediction.fake_probability,
    }


frontend_dist = WEBAPP_ROOT / "frontend" / "dist"
if frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
