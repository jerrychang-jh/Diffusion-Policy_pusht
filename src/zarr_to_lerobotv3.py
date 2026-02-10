import math
import json
import shutil
from pathlib import Path

import blosc2
import numpy as np

from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def read_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


# def read_zarr_chunk(array_dir: Path, chunk_coords, zmeta: dict):
#     """
#     Minimal Zarr v2 chunk reader for Blosc-compressed chunks (your dataset uses blosc+lZ4).
#     Works without installing `zarr` / `numcodecs`.
#     """
#     chunk_name = ".".join(str(c) for c in chunk_coords)
#     chunk_path = array_dir / chunk_name
#     comp = chunk_path.read_bytes()
#     raw = blosc2.decompress(comp)

#     dtype = np.dtype(zmeta["dtype"])
#     chunks = zmeta["chunks"]
#     shape = zmeta["shape"]

#     # Edge chunks can be smaller than nominal chunk size.
#     actual = []
#     for dim, csz, idx in zip(shape, chunks, chunk_coords):
#         start = idx * csz
#         end = min(dim, start + csz)
#         actual.append(end - start)

#     arr = np.frombuffer(raw, dtype=dtype)
#     return arr.reshape(actual, order=zmeta.get("order", "C"))


def read_zarr_chunk(array_dir: Path, chunk_coords, zmeta: dict):
    """
    Zarr v2 chunk reader that handles edge chunks stored as full (nominal) chunks.
    """
    chunk_name = ".".join(str(c) for c in chunk_coords)
    chunk_path = array_dir / chunk_name
    comp = chunk_path.read_bytes()
    raw = blosc2.decompress(comp)

    dtype = np.dtype(zmeta["dtype"])
    chunks = list(zmeta["chunks"])
    shape = list(zmeta["shape"])
    order = zmeta.get("order", "C")

    # chunk size at the dataset boundary
    actual = []
    starts = []
    for dim, csz, idx in zip(shape, chunks, chunk_coords):
        start = idx * csz
        end = min(dim, start + csz)
        starts.append(start)
        actual.append(max(0, end - start))

    # How many elements are stored in this chunk buffer
    stored_elems = len(raw) // dtype.itemsize
    nominal_elems = int(np.prod(chunks))
    actual_elems = int(np.prod(actual))

    arr = np.frombuffer(raw, dtype=dtype)

    # stored as full nominal chunk
    if stored_elems == nominal_elems:
        blk = arr.reshape(chunks, order=order)
        slicer = tuple(slice(0, a) for a in actual)
        return blk[slicer]

    # stored as truncated edge chunk
    if stored_elems == actual_elems:
        return arr.reshape(actual, order=order)

    raise ValueError(
        f"Unexpected chunk size for {chunk_path}:\n"
        f"stored_elems={stored_elems}, nominal_elems={nominal_elems}, actual_elems={actual_elems}\n"
        f"dtype={dtype}, chunks={chunks}, actual={actual}, coords={chunk_coords}"
    )


def iter_axis0_chunks(array_dir: Path, zmeta: dict):
    """
    Yield consecutive chunks along axis-0.
    Assumes non-axis0 dims are single-chunk (true for your data: img/state/action/keypoint/n_contacts).
    """
    chunks0 = zmeta["chunks"][0]
    n0 = zmeta["shape"][0]
    n_chunks0 = math.ceil(n0 / chunks0)

    other_chunk_counts = [
        math.ceil(s / c) for s, c in zip(zmeta["shape"][1:], zmeta["chunks"][1:])
    ]
    if any(x != 1 for x in other_chunk_counts):
        raise ValueError(
            f"Non-axis0 chunking not supported by this simple iterator: {other_chunk_counts}"
        )

    for i in range(n_chunks0):
        coords = (i,) + tuple(0 for _ in other_chunk_counts)
        yield i, read_zarr_chunk(array_dir, coords, zmeta)


def main(
    zarr_root: str,
    out_repo_id: str = "local/pusht_cchi_v1_lerobotv3",
    robot_type: str = "pusht",
    fps: int = 10,
):
    zarr_root = Path(zarr_root)

    # ---- Load zarr metadata
    img_dir = zarr_root / "data" / "img"
    action_dir = zarr_root / "data" / "action"
    state_dir = zarr_root / "data" / "state"
    keypoint_dir = zarr_root / "data" / "keypoint"
    n_contacts_dir = zarr_root / "data" / "n_contacts"
    episode_ends_dir = zarr_root / "meta" / "episode_ends"

    img_meta = read_json(img_dir / ".zarray")
    action_meta = read_json(action_dir / ".zarray")
    state_meta = read_json(state_dir / ".zarray")
    keypoint_meta = read_json(keypoint_dir / ".zarray")
    n_contacts_meta = read_json(n_contacts_dir / ".zarray")
    episode_ends_meta = read_json(episode_ends_dir / ".zarray")

    # ---- Read episode ends (1 chunk)
    episode_ends = read_zarr_chunk(episode_ends_dir, (0,), episode_ends_meta).astype(np.int64)
    total_frames = int(img_meta["shape"][0])
    assert int(episode_ends[-1]) == total_frames, "episode_ends last value must match total frames"

    # ---- Prepare output location
    out_path = HF_LEROBOT_HOME / out_repo_id
    if out_path.exists():
        shutil.rmtree(out_path)

    # ---- Create LeRobot dataset
    dataset = LeRobotDataset.create(
        repo_id=out_repo_id,
        robot_type=robot_type,
        fps=fps,
        features={
            "image": {
                "dtype": "image",
                "shape": (96, 96, 3),
                "names": ["height", "width", "channel"],
            },
            "robot_state": {"dtype": "float32", "shape": (5,), "names": ["robot_state"]},
            "action": {"dtype": "float32", "shape": (2,), "names": ["action"]},
            "keypoint": {"dtype": "float32", "shape": (9, 2), "names": ["keypoint"]},
            "n_contacts": {"dtype": "float32", "shape": (1,), "names": ["n_contacts"]},
            # "task": {"dtype": "string", "shape": (), "names": ["task"]},
        },
        image_writer_threads=4,
        image_writer_processes=1,
    )

    # ---- Stream all modalities chunk-by-chunk
    img_chunks = iter_axis0_chunks(img_dir, img_meta)
    action_chunks = iter_axis0_chunks(action_dir, action_meta)
    state_chunks = iter_axis0_chunks(state_dir, state_meta)
    keypoint_chunks = iter_axis0_chunks(keypoint_dir, keypoint_meta)
    n_contacts_chunks = iter_axis0_chunks(n_contacts_dir, n_contacts_meta)

    # Episode slicing bookkeeping
    ep_end_list = episode_ends.tolist()
    ep_idx = 0
    ep_start = 0
    ep_end = int(ep_end_list[ep_idx])

    global_t = 0

    for (ci, img_blk), (_, act_blk), (_, st_blk), (_, kp_blk), (_, nc_blk) in zip(
        img_chunks, action_chunks, state_chunks, keypoint_chunks, n_contacts_chunks
    ):
        # Iterate frames inside this block
        block_len = img_blk.shape[0]
        for j in range(block_len):
            if global_t >= total_frames:
                break

            # Convert image: stored as float32 0..255 → uint8
            img = img_blk[j]
            if img.dtype != np.uint8:
                img_u8 = np.clip(img, 0, 255).astype(np.uint8)
            else:
                img_u8 = img

            frame = {
                "image": img_u8,
                "robot_state": st_blk[j].astype(np.float32),
                "action": act_blk[j].astype(np.float32),
                "keypoint": kp_blk[j].astype(np.float32),
                "n_contacts": nc_blk[j].astype(np.float32),
                "task": "pusht_cchi_v1",
            }
            dataset.add_frame(frame)

            global_t += 1

            if global_t == ep_end:
                dataset.save_episode()
                ep_idx += 1
                ep_start = ep_end
                if ep_idx < len(ep_end_list):
                    ep_end = int(ep_end_list[ep_idx])

    print(f"Done. Wrote LeRobot dataset to: {out_path}")


if __name__ == "__main__":
    import sys

    zarr_root = sys.argv[1]
    main(zarr_root)
