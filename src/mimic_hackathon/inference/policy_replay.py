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

from mimic_hackathon.inference.policy_replay_helpers_both import (
    load_zarr_file_abs,
)


KEY_CHECKPOINT_PATH = "CHECKPOINT_PATH"
KEY_POLICY_PLAYER = "POLICY_PLAYER"


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

data = {}


def load_model_checkpoint() -> None:
    logger.info("Reading zarr file and preparing policy...")
    data["WAVE"] = load_zarr_file_abs(
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


HAS_INIT = False
WAVE_COUNTER = 0
N_ACTIONS = 15

print(data)


@app.get("/test")
async def predict() -> JSONResponse:
    """Get action prediction from the model given an observation."""
    global WAVE_COUNTER, N_ACTIONS, HAS_INIT

    # if not HAS_INIT:
    #     HAS_INIT = True
    #     init_config = data["WAVE"]["initial_config"]
    #     print(init_config.shape)
    #     assert init_config.shape == (25, )
    #     return {"actions": serialize([init_config for _ in range(N_ACTIONS)])}

    # if WAVE_COUNTER < len(data["WAVE"]["actions"]) - N_ACTIONS:
    #     actions = data["WAVE"]["actions"][WAVE_COUNTER: WAVE_COUNTER + N_ACTIONS]
    #     WAVE_COUNTER += N_ACTIONS

    #     for a in actions:
    #         print(a.shape)
    #     assert all([a.shape == (25, ) for a in actions ])
    #     print(actions.shape)
    #     return {"actions": serialize(actions)}
    #     # return JSONResponse(content=None)
    # else:
    #     return JSONResponse(content=None)
    if not HAS_INIT:
        HAS_INIT = True
        init_config = data["WAVE"]["initial_config"]
        print(init_config.shape)
        assert init_config.shape == (25,)
        return {"actions": serialize([init_config for _ in range(N_ACTIONS)])}

    if WAVE_COUNTER < len(data["WAVE"]["actions"]) - N_ACTIONS:
        actions = data["WAVE"]["actions"][WAVE_COUNTER : WAVE_COUNTER + N_ACTIONS]

        # extract the actions from the model output.
        ee_pos = actions[:, :3]
        ee_rot_6d = actions[:, 3:9].reshape(N_ACTIONS, 2, 3)
        hand_joints = actions[:, 9:]

        missing_row = np.cross(ee_rot_6d[:, 0], ee_rot_6d[:, 1])
        ee_rot = np.concatenate([ee_rot_6d, missing_row[:, None]], axis=1)

        ee_pose_ref = np.tile(np.eye(4), (N_ACTIONS, 1)).reshape(N_ACTIONS, 4, 4)
        ee_pose_ref[:, :3, :3] = ee_rot
        ee_pose_ref[:, :3, 3] = ee_pos

        for i, ee_pose in enumerate(ee_pose_ref):
            print(ee_pose)
            print(data["WAVE"]["ee_pose"][WAVE_COUNTER + i])
            assert np.equal(
                ee_pose, data["WAVE"]["ee_pose"][WAVE_COUNTER + i]
            ), "Don't match"

        WAVE_COUNTER += N_ACTIONS

        for a in actions:
            print(a.shape)
        assert all([a.shape == (25,) for a in actions])
        print(actions.shape)
        return {"actions": serialize(actions)}
        # return JSONResponse(content=None)
    else:
        return JSONResponse(content=None)


@app.post("/predict")
async def predict(observation: InputData) -> JSONResponse:
    """Get action prediction from the model given an observation."""
    global WAVE_COUNTER, N_ACTIONS, HAS_INIT

    obs = deserialize_numpy(observation.data)
    # print(f"got obs: {obs}")
    # pretty_print_dict(obs)

    if not HAS_INIT:
        HAS_INIT = True
        init_config = data["WAVE"]["initial_config"]
        print(init_config.shape)
        assert init_config.shape == (25,)
        return {"actions": serialize([init_config for _ in range(N_ACTIONS)])}

    if WAVE_COUNTER < len(data["WAVE"]["actions"]) - N_ACTIONS:
        actions = data["WAVE"]["actions"][WAVE_COUNTER : WAVE_COUNTER + N_ACTIONS]

        # extract the actions from the model output.
        ee_pos = actions[:, :3]
        ee_rot_6d = actions[:, 3:9].reshape(N_ACTIONS, 2, 3)
        hand_joints = actions[:, 9:]

        missing_row = np.cross(ee_rot_6d[:, 0], ee_rot_6d[:, 1])
        ee_rot = np.concatenate([ee_rot_6d, missing_row[:, None]], axis=1)

        ee_pose_ref = np.tile(np.eye(4), (N_ACTIONS, 1)).reshape(N_ACTIONS, 4, 4)
        ee_pose_ref[:, :3, :3] = ee_rot
        ee_pose_ref[:, :3, 3] = ee_pos

        for i, ee_pose in enumerate(ee_pose_ref):
            assert np.equal(
                ee_pose, data["WAVE"]["ee_pose"][WAVE_COUNTER + i]
            ), "Don't match"

        WAVE_COUNTER += N_ACTIONS

        for a in actions:
            print(a.shape)
        assert all([a.shape == (25,) for a in actions])
        print(actions.shape)
        return {"actions": serialize(actions)}
        # return JSONResponse(content=None)
    else:
        return JSONResponse(content=None)


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
