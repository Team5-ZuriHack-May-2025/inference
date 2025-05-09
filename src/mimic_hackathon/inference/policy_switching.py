import openai
import numpy as np
from typing import Dict, List, Any
import time

class PolicySwitchingController:
    def __init__(
        self,
        api_key: str,
        policies: Dict[str, Any],
        model: str = "gpt-4o",
        switch_threshold: float = 0.7
    ):
        """
        Initialize a policy switching controller.
        
        Args:
            api_key: OpenAI API key
            policies: Dictionary mapping policy names to policy objects
            model: OpenAI model to use for switching decisions
            switch_threshold: Confidence threshold for switching
        """
        openai.api_key = api_key
        self.policies = policies
        self.model = model
        self.switch_threshold = switch_threshold
        self.current_policy = None
        self.history = []
        
    def get_state_description(self, observation: Dict[str, np.ndarray]) -> str:
        """Convert robotics observation to a text description for the LLM."""
        # This is a simplified example - real implementation would depend on observation space
        description = "Current robot state: "
        
        if "joint_positions" in observation:
            joint_pos = observation["joint_positions"]
            description += f"Hand joints at {joint_pos.round(2).tolist()}. "
            
        if "object_position" in observation:
            obj_pos = observation["object_position"]
            description += f"Object at position {obj_pos.round(2).tolist()}. "
            
        if "tactile" in observation:
            contact = np.any(observation["tactile"] > 0.5)
            description += f"Contact detected: {contact}. "
            
        return description
    
    def decide_policy(self, observation: Dict[str, np.ndarray], task_description: str) -> str:
        """Use OpenAI API to determine which policy to use."""
        state_description = self.get_state_description(observation)
        
        messages = [
            {"role": "system", "content": (
                "You are a robotics controller that decides which specialized policy to use. "
                "Available policies: " + ", ".join(self.policies.keys()) + ". "
                "Respond only with the exact name of the policy to use and a confidence score "
                "between 0 and 1, in the format: 'policy_name:confidence'."
            )},
            {"role": "user", "content": (
                f"Task: {task_description}\n\n"
                f"State: {state_description}\n\n"
                f"Which policy should be used now?"
            )}
        ]
        
        # Add recent history for context
        for entry in self.history[-3:]:
            messages.append({"role": "assistant" if entry["is_response"] else "user", 
                             "content": entry["content"]})
        
        response = openai.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=20,
            temperature=0.2,
        )
        
        response_text = response.choices[0].message.content.strip()
        self.history.append({"content": response_text, "is_response": True})
        
        try:
            policy_name, confidence = response_text.split(":")
            policy_name = policy_name.strip()
            confidence = float(confidence.strip())
            
            if policy_name in self.policies and confidence >= self.switch_threshold:
                return policy_name
            else:
                return self.current_policy or list(self.policies.keys())[0]
        except:
            # Fallback if parsing fails
            return self.current_policy or list(self.policies.keys())[0]
    
    def get_action(self, observation: Dict[str, np.ndarray], task_description: str) -> np.ndarray:
        """Get action from the appropriate policy based on current state."""
        policy_name = self.decide_policy(observation, task_description)
        
        # Log policy switch if it occurred
        if self.current_policy != policy_name:
            print(f"Switching from {self.current_policy} to {policy_name}")
            self.current_policy = policy_name
        
        # Get action from the selected policy
        selected_policy = self.policies[policy_name]
        return selected_policy.predict(observation)
    

    # Example specialized policies
class GraspPolicy:
    def predict(self, obs):
        # Implementation for precision grasping
        return np.array([0.1, 0.2, 0.3, 0.4])  # Example action

class ManipulatePolicy:
    def predict(self, obs):
        # Implementation for in-hand manipulation
        return np.array([0.5, 0.4, 0.3, 0.2])  # Example action

class PlacePolicy:
    def predict(self, obs):
        # Implementation for precise placement
        return np.array([0.7, 0.6, 0.5, 0.4])  # Example action

# Create controller with specialized policies
controller = PolicySwitchingController(
    api_key="your_openai_api_key",
    policies={
        "grasp": GraspPolicy(),
        "manipulate": ManipulatePolicy(),
        "place": PlacePolicy()
    }
)

# Use in a control loop
task = "Pick up the red cube, rotate it 90 degrees, and place it on the blue target"
for _ in range(100):
    # Get current observation from robot
    observation = {
        "joint_positions": np.random.rand(9),
        "object_position": np.random.rand(3),
        "tactile": np.random.rand(6)
    }
    
    # Get action using appropriate policy
    action = controller.get_action(observation, task)
    
    # Execute action on robot
    # robot.execute(action)
    
    time.sleep(0.1)  # Control frequency