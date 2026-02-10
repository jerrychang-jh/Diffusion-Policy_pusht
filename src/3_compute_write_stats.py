# compute_and_write_stats.py
import json
from pathlib import Path
import numpy as np
import math
import shutil
import sys
from collections import defaultdict
from tqdm import trange

# try to import lerobot
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except Exception as e:
    print("Failed to import LeRobotDataset:", e)
    raise

# try cv2 for video frame sampling (optional)
try:
    import cv2
    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False

ROOT = Path("Dataset/lerobot_v3/pusht_cchi_v1_lerobotv3")
REPO_ID = "local/pust_cchi_v1_lerobotv3"   # not used except dataset init
N_SAMPLE_EPISODES = 300        # number of episodes to sample (tune down if slow)
N_FRAMES_PER_EP = 5            # frames per episode to sample for images
VIDEO_KEY_PREFIX = "observation.images.img"  # adjust if needed

print("Loading dataset...")
ds = LeRobotDataset(repo_id="local/pust_cchi_v1_lerobotv3", root=str(ROOT))

n_total = len(ds)
n_samples = min(N_SAMPLE_EPISODES, n_total)
print(f"Dataset length: {n_total}. Sampling {n_samples} episodes.")

# choose uniform indices to sample
if n_samples <= 0:
    print("No episodes found. Exiting.")
    sys.exit(1)

indices = np.linspace(0, n_total-1, n_samples, dtype=int).tolist()

# aggregator per key: running mean/std (Welford's)
class RunningStat:
    def __init__(self):
        self.n = 0
        self.mean = None
        self.M2 = None

    def update_batch(self, arr):
        # arr: numpy array, shape (N, ...); flatten sample axis
        arr = np.asarray(arr)
        if arr.size == 0:
            return
        # collapse all dims except channel/feature dims at the end if needed
        if arr.ndim > 1:
            # treat each sample as an independent vector
            flat = arr.reshape(arr.shape[0], -1).astype(np.float64)
        else:
            flat = arr.reshape(arr.shape[0], -1).astype(np.float64)
        # compute per-dimension stats across axis 0
        batch_n = flat.shape[0]
        batch_mean = flat.mean(axis=0)
        batch_M2 = ((flat - batch_mean)**2).sum(axis=0)

        if self.n == 0:
            self.n = batch_n
            self.mean = batch_mean
            self.M2 = batch_M2
        else:
            # combine
            delta = batch_mean - self.mean
            total_n = self.n + batch_n
            self.M2 = self.M2 + batch_M2 + (delta**2) * (self.n * batch_n / total_n)
            self.mean = (self.mean * self.n + batch_mean * batch_n) / total_n
            self.n = total_n

    def finalize(self):
        if self.n < 2:
            var = np.zeros_like(self.mean)
        else:
            var = self.M2 / self.n
        std = np.sqrt(var)
        return self.mean, std

# find keys by inspecting a single sample
sample_item = ds[indices[0]]
available_keys = list(sample_item.keys())
print("Available keys in one sample:", available_keys)

# choose numeric keys to compute stats for (exclude large episodic metadata)
numeric_keys = []
image_keys = []
for k in available_keys:
    v = sample_item[k]
    # simple heuristics: if it's an ndarray/tensor-like
    if hasattr(v, "shape"):
        if v.ndim >= 3 and (k.startswith("observation") or "image" in k or "img" in k):
            image_keys.append(k)
        else:
            numeric_keys.append(k)

print("Numeric keys:", numeric_keys)
print("Image keys:", image_keys)

stats = {}
# compute numeric keys stats
for k in numeric_keys:
    rs = RunningStat()
    for idx in indices:
        item = ds[idx]
        arr = item[k]
        # convert to numpy and ensure batch dimension if missing
        a = np.asarray(arr)
        # if it's a single sample (no leading batch dim), add batch dim
        if a.ndim == 0:
            a = a.reshape(1)
        elif a.ndim >= 1 and a.shape[0] != 1 and len(indices) == 1:
            # leave as is
            pass
        # ensure leading sample axis
        if a.ndim == 1:
            a = a.reshape(1, -1)
        elif a.ndim >= 2 and a.shape[0] != len(indices):
            # treat current sample as single sample
            a = a.reshape(1, -1)
        rs.update_batch(a)
    mean, std = rs.finalize()
    stats[k] = {
        "mean": mean.tolist() if mean is not None else None,
        "std": std.tolist() if std is not None else None,
    }
    print(f"Computed stats for {k}: mean_len={len(stats[k]['mean']) if stats[k]['mean'] is not None else 0}")

# compute image stats (sample frames if videos) — use ds to return frames if possible
for k in image_keys:
    rs = RunningStat()
    for idx in indices:
        item = ds[idx]
        try:
            img = item[k]   # may be (T,H,W,C) or (H,W,C)
            arr = np.asarray(img)
            # if arr is 3D with time first, sample frames
            if arr.ndim == 4:
                # arr: (T,H,W,C) -> sample up to N_FRAMES_PER_EP frames
                frames_idx = np.linspace(0, arr.shape[0]-1, min(arr.shape[0], 3), dtype=int)
                sample_frames = arr[frames_idx]
            elif arr.ndim == 3:
                sample_frames = arr[np.newaxis, ...]
            else:
                sample_frames = arr.reshape(1, *arr.shape)
            # flatten sample axis
            # convert uint8 -> [0,1]
            if sample_frames.dtype == np.uint8:
                sample_frames = sample_frames.astype(np.float32) / 255.0
            # reshape to (Nframes, channels, H, W) if needed -> we handle generic shapes in RunningStat
            # collapse spatial dims; RunningStat handles that
            rs.update_batch(sample_frames)
        except Exception as e:
            # if ds can't return frames (e.g., only video metadata), try to sample mp4 directly using cv2
            if HAVE_CV2:
                # attempt to find mp4 for episode by reading dataset metadata
                try:
                    # LeRobot exposes ep metadata via ds.meta possibly, but to be robust, fallback to a default strategy
                    # Try to get chunk_index/file_index from item (if present)
                    ep = item.get("episode_index", None) or item.get("ep_idx", None)
                except Exception:
                    ep = None
                # we skip detailed mp4 sampling in fallback to keep script robust
                continue
            else:
                continue
    mean, std = rs.finalize()
    if mean is None:
        # fallback default
        print(f"Could not compute image stats for {k}. Falling back to mean=0.5,std=0.5")
        stats[k] = {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}
    else:
        # mean array corresponds to flattened per-channel mean; we need to reduce per-channel
        # assume last dimension is channel size, so reshape back
        # heuristic: if mean length divisible by 3 -> compute channel means by summing channel positions
        mn = np.asarray(mean)
        sd = np.asarray(std)
        # try to infer channels by dividing by H*W using sample image size
        sample_img = np.asarray(sample_item[image_keys[0]])
        if sample_img.ndim == 4:
            # T,H,W,C
            _, H, W, C = sample_img.shape
        elif sample_img.ndim == 3:
            H, W, C = sample_img.shape
        else:
            H=W=C=1
        if mn.size == H*W*C and C>1:
            mn = mn.reshape(-1, C).mean(axis=0)
            sd = sd.reshape(-1, C).mean(axis=0)
        else:
            # if can't infer shape, take first 3 elements
            mn = mn[:3]
            sd = sd[:3]
        stats[k] = {"mean": [float(x) for x in mn.tolist()], "std": [float(x) for x in sd.tolist()]}
        print(f"Image stats for {k}: mean(len)={len(stats[k]['mean'])}")

# write stats to ROOT/meta/stats.json (back up if exists)
out = ROOT / "meta" / "stats.json"
bak = out.with_suffix(".json.bak")
if out.exists() and not bak.exists():
    shutil.copy2(out, bak)
with open(out, "w") as f:
    json.dump(stats, f, indent=2)
print("Wrote stats to:", out)
print("Done.")
