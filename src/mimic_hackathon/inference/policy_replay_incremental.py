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
import numpy as np

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

from mimic_hackathon.inference.policy_replay_helpers import (
    load_zarr_file,
)


KEY_CHECKPOINT_PATH = "CHECKPOINT_PATH"
KEY_POLICY_PLAYER = "POLICY_PLAYER"


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

data = {}


def load_model_checkpoint() -> None:
    logger.info("Reading zarr file and preparing policy...")
    data["WAVE"] = load_zarr_file(
        "/home/sjhu/Repositories/ZuriHack/datasets/kitchen_data_wave_2025_05_10_11_32_41.zarr"
    )


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
    formatted_dict = {k: {"shape": v.shape} for k, v in d.items()}
    pprint.pprint(formatted_dict, width=80)


HAS_CENTERED = False
WAVE_COUNTER = 0
N_ACTIONS = 15


@app.get("/test")
async def predict() -> JSONResponse:
    """Get action prediction from the model given an observation."""
    global WAVE_COUNTER, N_ACTIONS, HAS_CENTERED

    if not HAS_CENTERED:
        HAS_CENTERED = True
        zeroing = [
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ] + [0.0] * 16
        return {"actions": serialize([zeroing for _ in range(N_ACTIONS)])}

    if WAVE_COUNTER < len(data["WAVE"]["actions"]) - N_ACTIONS:
        actions = data["WAVE"]["actions"][WAVE_COUNTER : WAVE_COUNTER + N_ACTIONS]
        WAVE_COUNTER += N_ACTIONS

        # extract the actions from the model output.
        ee_pos = actions[:, :3]
        ee_rot_6d = actions[:, 3:9].reshape(N_ACTIONS, 2, 3)

        missing_row = np.cross(ee_rot_6d[:, 0], ee_rot_6d[:, 1])
        ee_rot = np.concatenate([ee_rot_6d, missing_row[:, None]], axis=1)

        ee_pose_ref = np.tile(np.eye(4), (N_ACTIONS, 1)).reshape(N_ACTIONS, 4, 4)
        ee_pose_ref[:, :3, :3] = ee_rot
        ee_pose_ref[:, :3, 3] = ee_pos

        # print(ee_pose_ref.shape)

        chain = np.empty_like(ee_pose_ref)
        # ee_pose_ref has shape (n_actions, 4, 4)
        n = ee_pose_ref.shape[0]
        chain = np.empty_like(ee_pose_ref)
        chain[0] = ee_pose_ref[0]
        for i in range(1, N_ACTIONS):
            chain[i] = chain[i - 1] @ ee_pose_ref[i]

        actions[:, :3] = chain[:, :3, 3]
        actions[:, 3:9] = chain[:, :2, :3].reshape(N_ACTIONS, -1)

        # # opt 1: repeat same action 10 times
        # WAVE_COUNTER += 1
        # chunks = [action for _ in range(10)]

        return {"actions": serialize(actions)}
        # return JSONResponse(content=None)
    else:
        return JSONResponse(content=None)
    # actions = data[KEY_POLICY_PLAYER].step(obs) if KEY_POLICY_PLAYER in data else None
    # if actions is not None:
    #     return {"actions": serialize(actions)}
    # else:
    #     return JSONResponse(content=None)


@app.post("/predict")
async def predict(observation: InputData) -> JSONResponse:
    """Get action prediction from the model given an observation."""
    global WAVE_COUNTER, N_ACTIONS, HAS_CENTERED

    obs = deserialize_numpy(observation.data)
    # print(f"got obs: {obs}")
    pretty_print_dict(obs)
    if not HAS_CENTERED:
        HAS_CENTERED = True
        zeroing = [
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ] + [0.0] * 16
        return {"actions": serialize([zeroing for _ in range(N_ACTIONS)])}

    if WAVE_COUNTER < len(data["WAVE"]["actions"]) - N_ACTIONS:
        actions = data["WAVE"]["actions"][WAVE_COUNTER : WAVE_COUNTER + N_ACTIONS]
        WAVE_COUNTER += N_ACTIONS

        # extract the actions from the model output.
        ee_pos = actions[:, :3]
        ee_rot_6d = actions[:, 3:9].reshape(N_ACTIONS, 2, 3)

        missing_row = np.cross(ee_rot_6d[:, 0], ee_rot_6d[:, 1])
        ee_rot = np.concatenate([ee_rot_6d, missing_row[:, None]], axis=1)

        ee_pose_ref = np.tile(np.eye(4), (N_ACTIONS, 1)).reshape(N_ACTIONS, 4, 4)
        ee_pose_ref[:, :3, :3] = ee_rot
        ee_pose_ref[:, :3, 3] = ee_pos

        # print(ee_pose_ref.shape)

        chain = np.empty_like(ee_pose_ref)
        # ee_pose_ref has shape (n_actions, 4, 4)
        n = ee_pose_ref.shape[0]
        chain = np.empty_like(ee_pose_ref)
        chain[0] = ee_pose_ref[0]
        for i in range(1, N_ACTIONS):
            chain[i] = chain[i - 1] @ ee_pose_ref[i]

        actions[:, :3] = chain[:, :3, 3]
        actions[:, 3:9] = chain[:, :2, :3].reshape(N_ACTIONS, -1)

        # # opt 1: repeat same action 10 times
        # WAVE_COUNTER += 1
        # chunks = [action for _ in range(10)]

        print(actions)

        return {"actions": serialize(actions)}
        # return JSONResponse(content=None)
    else:
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
