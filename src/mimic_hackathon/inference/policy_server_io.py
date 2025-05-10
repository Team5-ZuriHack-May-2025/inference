"""Policy server (template). Serves a robot policy/model as a FastAPI application
using Uvicorn as ASGI server to serve it and makes it available at
http://<server-ip>:<server-port>/docs

Run infernece server:
python policy_server.py \
    --ip <server-ip> \
    --port <server-port> \
    --checkpoint-path /path/to/checkpoint
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import tyro
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pprint

from mimic_hackathon.inference.serialization import (
    InputData,
    deserialize_numpy,
    serialize,
)


KEY_CHECKPOINT_PATH = "CHECKPOINT_PATH"
KEY_POLICY_PLAYER = "POLICY_PLAYER"


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

data = {}


def load_model_checkpoint() -> None:
    logger.info("IO test skipping checkpoint loading.")


@asynccontextmanager
async def lifespan(app: FastAPI) -> None:  # type: ignore
    """Asynchronous context manager for the lifespan of the FastAPI app.

    - Entering the context: Loads the model.
    - Exiting the context: Handles cleanup tasks.

    Args:
        app (FastAPI): FastAPI app instance for which the lifespan is being managed.
    """
    load_model_checkpoint()
    yield
    data.clear()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root() -> JSONResponse:
    """
    Return debugging information about the server.
    """
    content = {
        "Python": sys.version,
        "Checkpoint": data[KEY_CHECKPOINT_PATH].name,
    }
    return JSONResponse(content)


def pretty_print_dict(d):
    formatted_dict = {k: {"shape": v.shape, "data": v.tolist()} for k, v in d.items()}
    pprint.pprint(formatted_dict, width=80)


@app.post("/predict")
async def predict(observation: InputData) -> JSONResponse:
    """Get action prediction from the model given an observation."""
    obs = deserialize_numpy(observation.data)
    print(f"got obs: {obs}")
    pretty_print_dict(obs)
    return JSONResponse(content=None)
    # actions = data[KEY_POLICY_PLAYER].step(obs) if KEY_POLICY_PLAYER in data else None
    # if actions is not None:
    #     return {"actions": serialize(actions)}
    # else:
    #     return JSONResponse(content=None)


def main(
    ip: str,
    port: int,
    checkpoint_path: Path,
):
    # Set checkpoint path.
    if not checkpoint_path.is_file():
        logger.error(f"{checkpoint_path} doesn't exist.")
        sys.exit(1)
    data[KEY_CHECKPOINT_PATH] = checkpoint_path
    logger.info(f"Checkpoint path set: {data[KEY_CHECKPOINT_PATH]}")
    # Run the application.
    uvicorn.run(
        app,
        host=ip,
        port=port,
        loop="asyncio",
    )


if __name__ == "__main__":
    tyro.cli(main)
