import numpy as np
import zarr
from mimic_hackathon.inference.policy_replay_helpers import (
    complete_rotation_matrices_2_3,
)
from scipy.spatial.transform import Rotation as R

STEPS_FOR_CHUNK = 48


def build_goto_policy(goal: np.ndarray):
    """
    Build the goto policy for the agent given a goal.
    """

    def predict(obs: dict):
        # we keep the hand fix while moving
        current_hand = obs["robot0_hand_joints_lowdim"][-1]
        hand_chunks = np.tile(current_hand, (STEPS_FOR_CHUNK, 16))
        current_pose = obs["robot0_eef_pose_lowdim"][-1]
        current_translation = current_pose[:3, 3]
        current_rotation = current_pose[:2, :3]

        # extract goal information
        goal_translation = goal[:3, 3]
        goal_rotation = goal[:2, :3]

        # build full translation trajectory
        translation_step = (goal_translation - current_translation) / STEPS_FOR_CHUNK
        translation_chunks = np.array(
            [(i + 1) * translation_step for i in range(STEPS_FOR_CHUNK)]
        )

        print("translation chunks")
        print(translation_chunks)

        # build rotations trajectory
        goal_rot_object = R.from_matrix(
            complete_rotation_matrices_2_3([goal_rotation])[0]
        )
        current_rot_object = R.from_matrix(
            complete_rotation_matrices_2_3([current_rotation])[0]
        )
        required_rotation_matrix = goal_rot_object.inv() * current_rot_object
        degrees = required_rotation_matrix.as_euler("xyz")
        rotation_step = degrees / STEPS_FOR_CHUNK
        rotation_chunks_degree = [i + 1 * rotation_step for i in range(STEPS_FOR_CHUNK)]
        rotation_chunks = np.array(
            [
                R.from_euler("xyz", deg).as_matrix()[:2, :].ravel()
                for deg in rotation_chunks_degree
            ]
        )

        print("rotation chunks")
        print(rotation_chunks)

        action = np.concatenate(
            (translation_chunks, rotation_chunks, hand_chunks), axis=1
        )

        print("action shape")
        print(action.shape)
        return action

    return predict


if __name__ == "__main__":
    root_group = zarr.open(
        "/home/mario/mimichack/inference/kitchen_data_wave_2025_05_10_11_32_41.zarr/kitchen_data_wave_2025_05_10_11_32_41.zarr",
        mode="r",
    )
    pose_dt = root_group["robot0_eef_pose_lowdim"][...]
    goal = pose_dt[120]
    current = pose_dt[0]
    print("building trajectory")
    print("from", current)
    print("to", goal)
    build_goto_policy(goal)(
        {
            "robot0_eef_pose_lowdim": [current],
            "robot0_hand_joints_lowdim": np.zeros(16),
        }
    )
