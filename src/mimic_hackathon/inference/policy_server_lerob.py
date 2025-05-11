"""
Policy server (multi-policy). Serves multiple robot policies/models as a FastAPI app
using Uvicorn. Available at http://<server-ip>:<server-port>/docs

Run inference server:
python policy_server.py \
    --ip <server-ip> \
    --port <server-port>
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import tyro
import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from datasets import load_dataset
from datetime import datetime

import numpy as np

from mimic_hackathon.inference.serialization import (
    InputData,
    deserialize_numpy,
    serialize,
)

KEY_AVAILABLE_POLICIES = "POLICIES"
KEY_ACTIVE_POLICY = "ACTIVE_POLICY"
data = {}

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

policies_info = {
    "green": {"dataset": "chengkunli/green_cup_pour", "start": 100, "end": 750},
    "blue": {"dataset": "chengkunli/blue_cup_pour", "start": 0, "end": 400},
    "red": {"dataset": "chengkunli/red_cup_pour", "start": 0, "end": 400},
    "yellow": {"dataset": "chengkunli/yellow_cup_pour", "start": 0, "end": 400},
}


# still policy does not move
class DefaultPolicy:
    def predict(obs):
        return None


class GoToPolicy:
    def __init__(self, goal: np.ndarray, cb: str):
        self.goal = goal
        self.callback = cb
        self.init_time = datetime.now()

    def predict(self, obs: dict):
        time_difference = (datetime.now() - self.init_time).total_seconds()
        if time_difference >= 4:
            data[KEY_ACTIVE_POLICY] = data[KEY_AVAILABLE_POLICIES][self.callback]
        return [self.goal]


class HardcodedPolicy:
    def __init__(self, dataset_name, start_t, end_t):
        self.dataset = load_dataset(dataset_name)
        self.start_t = start_t
        self.end_t = end_t

    def play(self, start_t, end_t):
        """
        Play the game from start_t to end_t
        """
        self.t = start_t
        while self.t < end_t:
            action = self.get_action(self.t)
            self.t += 1
            yield action

    def predict(self):
        """
        Predict the action from start_t to end_t
        """
        action = self.get_action(self.t)
        return action

    def get_action(self, t):
        return self.dataset["train"]["action"][t]

    def get_state(self, t=None):
        if not t:
            t = self.t
        return self.dataset["train"]["observation.state"][t]


def load_all_policies() -> None:
    available_policies = {}
    for name, path in policies_info.items():
        try:
            available_policies[name] = HardcodedPolicy(
                path["dataset"],
                path["start"],
                path["end"],
            )
        except Exception as e:
            logger.error(f"Failed to load policy '{name}': {e}")
            import traceback

            traceback.print_exc()
    data[KEY_AVAILABLE_POLICIES] = available_policies
    data[KEY_ACTIVE_POLICY] = DefaultPolicy()


@asynccontextmanager
async def lifespan(app: FastAPI) -> None:
    load_all_policies()
    yield
    data.clear()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict to frontend URL if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> JSONResponse:
    return JSONResponse(
        content={
            "Python": sys.version,
            "Loaded Policies": list(data.get(KEY_AVAILABLE_POLICIES, {}).keys()),
            "Active Policy": data.get(KEY_ACTIVE_POLICY, None),
        }
    )


@app.post("/set_policy")
async def set_policy(
    policy_name: str = Query(..., description="Name of the policy to set as active"),
) -> JSONResponse:
    """Set the active policy to use"""
    print(data[KEY_AVAILABLE_POLICIES])

    if policy_name not in data[KEY_AVAILABLE_POLICIES]:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"Policy '{policy_name}' not found. Available policies: {list(data[KEY_AVAILABLE_POLICIES].keys())}"
            },
        )

    wanted_policy = data[KEY_AVAILABLE_POLICIES][policy_name]
    data[KEY_ACTIVE_POLICY] = GoToPolicy(wanted_policy.get_state(), policy_name)
    logger.info(f"Active policy set to '{policy_name}'")
    return JSONResponse(content={"status": "success", "active_policy": policy_name})


@app.post("/predict")
async def predict(
    observation: InputData,
) -> JSONResponse:
    obs = deserialize_numpy(observation.data)
    actions = data[KEY_ACTIVE_POLICY].predict(obs)
    print("active policy", data[KEY_ACTIVE_POLICY])
    print("actions", actions)
    if actions is None:
        return JSONResponse(content=None)
    else:
        return JSONResponse(content={"actions": serialize(actions)})


def main(ip: str, port: int):
    uvicorn.run(app, host=ip, port=port, loop="asyncio")


if __name__ == "__main__":
    tyro.cli(main)
