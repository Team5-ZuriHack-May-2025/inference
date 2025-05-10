import zarr
import tyro
import numpy as np
from scipy.spatial.transform import Rotation as R


def complete_rotation_matrices_3_2(rotations):
    """Convert an array of (3,2) rotation matrices to full (3,3) matrices."""
    full_rotations = []

    for rot in rotations:
        # Extract first two columns as basis vectors
        v1, v2 = rot[:, 0], rot[:, 1]

        # Compute the third basis vector using cross product
        v3 = np.cross(v1, v2)

        # Stack to form a full (3×3) rotation matrix
        full_rot = np.column_stack((v1, v2, v3))
        full_rotations.append(full_rot)

    return np.array(full_rotations)


def complete_rotation_matrices_2_3(rotations):
    """Convert an array of (2,3) rotation matrices to full (3,3) matrices."""
    full_rotations = []

    for rot in rotations:
        # Extract first two rows as basis vectors
        v1, v2 = rot[0, :], rot[1, :]

        # Compute the third basis vector using cross product
        v3 = np.cross(v1, v2)

        # Stack to form a full (3×3) rotation matrix
        full_rot = np.row_stack((v1, v2, v3))
        full_rotations.append(full_rot)

    return np.array(full_rotations)


# get rotation diff between two 3x3 rotation matrices (a-b) and return the rotation matrix
def rotation_diff(a, b):
    ar, br = R.from_matrix(a), R.from_matrix(b)
    diff = ar.inv() * br
    return diff


def load_zarr_file(zarr_path: str):
    try:
        # Open the Zarr hierarchy (root group)
        # 'r' mode is for read-only
        root_group = zarr.open(zarr_path, mode="r")

        print(f"Successfully opened Zarr store: {zarr_path}\n")

        print(root_group.tree())

        # List of the channels you want to load
        channel_names = [
            "robot0_eef_pose_lowdim",
            "robot0_eef_pose_lowdim_timestamps",
            "robot0_eef_pose_ref_lowdim",
            "robot0_eef_pose_ref_lowdim_timestamps",
            "robot0_hand_joints_lowdim",
            "robot0_hand_joints_lowdim_timestamps",
        ]

        # Load each into memory as a NumPy array
        loaded_data = {}
        for name in channel_names:
            # this reads the entire zarr array into a numpy.ndarray
            loaded_data[name] = root_group[name][...]

        # Now you can unpack or inspect them:
        robot0_eef_pose_lowdim = loaded_data[
            "robot0_eef_pose_lowdim"
        ]  # shape (324, 4, 4), dtype float64
        eef_ts = loaded_data[
            "robot0_eef_pose_lowdim_timestamps"
        ]  # shape (324,),     dtype uint64
        robot0_eef_pose_ref_lowdim = loaded_data[
            "robot0_eef_pose_ref_lowdim"
        ]  # shape (324, 4, 4), dtype float64
        ref_ts = loaded_data[
            "robot0_eef_pose_ref_lowdim_timestamps"
        ]  # shape (324,),     dtype uint64
        hand_ds = loaded_data[
            "robot0_hand_joints_lowdim"
        ]  # shape (216, 16),  dtype float32
        hand_ts = loaded_data[
            "robot0_hand_joints_lowdim_timestamps"
        ]  # shape (216,),     dtype uint64

        matched_indices = np.array([np.abs(eef_ts - t).argmin() for t in hand_ts])

        # Sanity‐check: indices should be non‐decreasing (hand_ts assumed sorted)
        assert np.all(np.diff(matched_indices) >= 0), "Matched indices not monotonic!"

        # Extract the downsampled pose arrays + their timestamps
        pose_ds = robot0_eef_pose_lowdim[matched_indices]
        pose_ref_ds = robot0_eef_pose_ref_lowdim[matched_indices]
        pose_ds_ts = eef_ts[matched_indices]
        pose_ref_ds_ts = ref_ts[matched_indices]

        print("hand ds", hand_ds.shape, hand_ds.dtype)
        print("hand ts", hand_ts.shape, hand_ts.dtype)
        print("pose ds", pose_ds.shape, pose_ds.dtype)
        print("pose ts", pose_ds_ts.shape, pose_ds_ts.dtype)
        print("pose ref ds", pose_ref_ds.shape, pose_ref_ds.dtype)
        print("pose ref ts", pose_ref_ds_ts.shape, pose_ref_ds_ts.dtype)

        clip_length = hand_ts[-1] - hand_ts[0]

        # Double‐check timing errors
        # Double‐check timing errors
        time_errors = (
            np.abs(hand_ts.astype(np.int64) - pose_ds_ts.astype(np.int64)).astype(
                np.float32
            )
            / 1_000_000_000
        )
        print("Max time error (hand_ts - matched_eef_ts):", time_errors.max(), "sec")
        print("Mean time error:", time_errors.mean(), "sec")

        # extract tranlsations and rotations
        translations = pose_ds[:, :3, 3]
        translations_diff = np.diff(translations, axis=0)

        print("translations shape", translations.shape)
        print("translations diff shape", translations_diff.shape)

        # print("translations", translations)
        # print("translations_diff", translations_diff)

        # IMPORTANT: we are taking 2x3 as repr of rotation instead of 3x2 (same angles, but need consistent changes to code below)
        rotations = pose_ds[:, :2, :3]
        print("rotations shape", rotations.shape)
        # print("rotations", rotations)
        full_rotations = complete_rotation_matrices_2_3(rotations)
        print("rotations shape", full_rotations.shape)
        # print("rotations", full_rotations)
        # Convert to SciPy Rotation objects
        rotation_objects = R.from_matrix(full_rotations)

        # euler_angles = [rd.as_euler("xyz", degrees=True) for rd in rotation_objects]
        # print("Euler Angles rot (degrees):")
        # print(euler_angles[0], euler_angles[1])

        # Compute differences between consecutive rotations
        rotation_obj_diffs = [
            rotation_objects[i].inv() * rotation_objects[i + 1]
            for i in range(len(rotation_objects) - 1)
        ]

        # euler_angles = [rd.as_euler("xyz", degrees=True) for rd in rotation_obj_diffs]
        # print("Euler Angles rot diff (degrees):")
        # print(euler_angles)

        # Extract differences as rotation matrices or quaternions
        diff_matrices = np.array([diff.as_matrix() for diff in rotation_obj_diffs])

        print("diff_matrices shape", diff_matrices.shape)

        rotations_diff = diff_matrices[:, :2, :].reshape(diff_matrices.shape[0], -1)
        print("rotations_diff shape", rotations_diff.shape)
        # print("rotations_diff")
        # print(rotations_diff[0])
        # print("diff mat")
        # print(diff_matrices[0])
        print("actions diff shape")
        print(translations_diff.shape, "translations")
        print(rotations_diff.shape, "rotations")
        print(hand_ds[1:, :].shape, "hand pose")

        actions = np.concatenate(
            (translations_diff, rotations_diff, hand_ds[1:, :]), axis=1
        )
        print(actions.shape, "actions shape")
        print(clip_length, "clip_length")
        print(pose_ds[0], "initial arm pose")

        return {
            "actions": actions,
            "clip_length": clip_length,
            "initial_arm_pose": pose_ds[0],
        }

    except FileNotFoundError:
        print(f"Error: Zarr store not found at {zarr_path}. Please check the path.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    tyro.cli(load_zarr_file)
