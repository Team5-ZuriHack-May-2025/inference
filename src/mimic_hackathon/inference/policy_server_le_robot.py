import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
import os
import torch
import numpy as np

import tyro
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from pathlib import Path
path = Path("/Users/soumya/Desktop/zurich_builds/mimic_hackathon/src/mimic_hackathon/inference/serialization.py")
print(path.parent.absolute())

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


class ACTPolicyPlayer:
    """Player for ACT policy models that handles inference for dexterous hand control."""
    
    def __init__(self, checkpoint_path: Path):
        """Initialize the policy player with a checkpoint.
        
        Args:
            checkpoint_path: Path to the model checkpoint
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load the checkpoint
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load the model architecture
        model_cls = self._get_model_class()
        self.model = model_cls(**self.checkpoint.get("model_args", {}))
        
        # Load the model weights
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        
        # Extract metadata
        self.chunk_size = self.checkpoint.get("chunk_size", 16)
        self.action_dim = self.checkpoint.get("action_dim", 16)
        
        # Initialize action buffer for chunking
        self.action_buffer = []
        self.current_chunk_idx = 0
        
        logger.info(f"Loaded ACT policy with action dim: {self.action_dim}, chunk size: {self.chunk_size}")
    
    def _get_model_class(self):
        """Get the model class based on the checkpoint architecture."""
        # This would usually import the correct model class based on checkpoint metadata
        # For example, if the checkpoint contains a model_type field
        model_type = self.checkpoint.get("model_type", "act")
        
        if model_type == "act":
            # Import your actual ACT model implementation here
            from mimic_hackathon.models.act import ACTPolicy
            return ACTPolicy
        elif model_type == "dexformer":
            from mimic_hackathon.models.dexformer import DexformerPolicy
            return DexformerPolicy
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _process_observation(self, observation):
        """Process raw observation dictionary into model input format."""
        processed_obs = {}
        
        # Process images if present
        for key in observation:
            if key.endswith("_rgb") and isinstance(observation[key], np.ndarray):
                # Ensure correct shape and normalization for vision inputs
                if observation[key].ndim == 3:  # Handle single image
                    img = observation[key][None]  # Add batch dimension
                else:
                    img = observation[key]
                
                # Convert to torch tensor
                processed_obs[key] = torch.from_numpy(img).float().to(self.device)
            elif isinstance(observation[key], np.ndarray):
                # Handle other numpy arrays (joint states, etc.)
                processed_obs[key] = torch.from_numpy(observation[key]).float().to(self.device)
            else:
                # Pass through other data types
                processed_obs[key] = observation[key]
        
        return processed_obs
    
    def step(self, observation):
        """Generate actions for the given observation.
        
        Returns chunked actions according to the ACT policy design.
        """
        # Check if we have actions left in the buffer
        if self.action_buffer and self.current_chunk_idx < len(self.action_buffer):
            action = self.action_buffer[self.current_chunk_idx]
            self.current_chunk_idx += 1
            return action
        
        # Process new observation and generate a new chunk of actions
        with torch.no_grad():
            processed_obs = self._process_observation(observation)
            
            # Run the model inference
            model_output = self.model(processed_obs)
            
            # Extract action chunk from model output
            if isinstance(model_output, dict):
                # Some models return a dictionary with actions and other info
                action_chunk = model_output["actions"]
            elif isinstance(model_output, torch.Tensor):
                # Some models return the actions directly
                action_chunk = model_output
            else:
                raise ValueError(f"Unexpected model output type: {type(model_output)}")
            
            # Convert to numpy and store in buffer
            action_chunk = action_chunk.cpu().numpy()
            
            # Handle different action chunk shapes based on model architecture
            if action_chunk.ndim == 3:  # [batch, chunk_size, action_dim]
                self.action_buffer = [action_chunk[0, i] for i in range(action_chunk.shape[1])]
            elif action_chunk.ndim == 2:  # [chunk_size, action_dim]
                self.action_buffer = [action_chunk[i] for i in range(action_chunk.shape[0])]
            else:
                # Fallback for single-step models
                self.action_buffer = [action_chunk]
            
            # Reset index and return first action
            self.current_chunk_idx = 1  # Start at 1 since we're returning the first action
            return self.action_buffer[0]


def load_model_checkpoint() -> None:
    checkpoint_path = data[KEY_CHECKPOINT_PATH]
    logger.info(f"Loading {checkpoint_path.name} checkpoint ...")
    
    try:
        # Create the policy player with the checkpoint
        policy_player = ACTPolicyPlayer(checkpoint_path)
        data[KEY_POLICY_PLAYER] = policy_player
        logger.info(f"Checkpoint {checkpoint_path.name} loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        # Print detailed traceback
        import traceback
        traceback.print_exc()
        sys.exit(1)


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
        "Model Type": data[KEY_POLICY_PLAYER].checkpoint.get("model_type", "unknown"),
        "Action Dimension": data[KEY_POLICY_PLAYER].action_dim,
        "Chunk Size": data[KEY_POLICY_PLAYER].chunk_size,
        "Device": str(data[KEY_POLICY_PLAYER].device),
    }
    return JSONResponse(content)


@app.post("/predict")
async def predict(observation: InputData) -> JSONResponse:
    """Get action prediction from the model given an observation."""
    obs = deserialize_numpy(observation.data)
    actions = data[KEY_POLICY_PLAYER].step(obs) if KEY_POLICY_PLAYER in data else None
    if actions is not None:
        return JSONResponse(content={"actions": serialize(actions)})
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