from pathlib import Path
import re
import pandas as pd
import numpy as np
import shutil

ROOT = Path("Dataset/lerobot_v3/pusht_cchi_v1_lerobotv3")
episodes_root = ROOT / "meta" / "episodes"

VIDEO_KEY = "observation.images.img"
COL_NFRAMES = f"videos/{VIDEO_KEY}/num_frames"
COL_FROM_TS = f"videos/{VIDEO_KEY}/from_timestamp"
COL_TO_TS   = f"videos/{VIDEO_KEY}/to_timestamp"
COL_FPS     = f"videos/{VIDEO_KEY}/fps"

COL_FROM_IDX = "dataset_from_index"
COL_TO_IDX   = "dataset_to_index"

parquets = sorted(episodes_root.glob("chunk-*/**/*.parquet"))
if not parquets:
    raise RuntimeError(f"No parquet files found under {episodes_root}")

all_rows = []
row_sources = []

for pq in parquets:
    df = pd.read_parquet(pq)

    if "episode_index" in df.columns:
        order = np.argsort(df["episode_index"].to_numpy())
        df = df.iloc[order].reset_index(drop=True)

    # compute episode length T
    if COL_NFRAMES in df.columns and df[COL_NFRAMES].notna().all():
        T = df[COL_NFRAMES].astype(int).to_numpy()
    else:
        # fallback: (to-from)*fps
        if not all(c in df.columns for c in [COL_FROM_TS, COL_TO_TS, COL_FPS]):
            raise RuntimeError(
                f"{pq} missing episode length fields. Need either {COL_NFRAMES} "
                f"or ({COL_FROM_TS},{COL_TO_TS},{COL_FPS})."
            )
        dur = (df[COL_TO_TS] - df[COL_FROM_TS]).to_numpy(dtype=float)
        fps = df[COL_FPS].to_numpy(dtype=float)
        T = np.maximum(1, np.round(dur * fps).astype(int))

    # store rows
    all_rows.append(pd.DataFrame({
        "_pq": str(pq),
        "_row": np.arange(len(df), dtype=int),
        "_T": T,
        "_episode_index": df["episode_index"].to_numpy() if "episode_index" in df.columns else np.arange(len(df)),
    }))
    row_sources.append((pq, len(df)))

all_df = pd.concat(all_rows, ignore_index=True)


use_ep = all_df["_episode_index"].is_monotonic_increasing or all_df["_episode_index"].nunique() > 0.9 * len(all_df)
if use_ep:
    all_df = all_df.sort_values("_episode_index").reset_index(drop=True)
else:
    all_df = all_df.sort_values(["_pq", "_row"]).reset_index(drop=True)

# prefix sum
from_idx = np.zeros(len(all_df), dtype=int)
to_idx = np.zeros(len(all_df), dtype=int)
running = 0
for i, t in enumerate(all_df["_T"].to_numpy()):
    from_idx[i] = running
    running += int(t)
    to_idx[i] = running

all_df[COL_FROM_IDX] = from_idx
all_df[COL_TO_IDX] = to_idx

# Write back per parquet
grouped = all_df.groupby("_pq", sort=False)
patched = 0

for pq_str, g in grouped:
    pq = Path(pq_str)
    df = pd.read_parquet(pq)

    # If episode_index exists, align by sorting then writing then unsorting
    if "episode_index" in df.columns:
        order = np.argsort(df["episode_index"].to_numpy())
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))

        df_sorted = df.iloc[order].reset_index(drop=True)
        df_sorted[COL_FROM_IDX] = g[COL_FROM_IDX].to_numpy()
        df_sorted[COL_TO_IDX] = g[COL_TO_IDX].to_numpy()

        # restore original row order
        df_out = df_sorted.iloc[inv].reset_index(drop=True)
    else:
        if len(df) != len(g):
            raise RuntimeError(f"Row count mismatch for {pq}")
        df_out = df.copy()
        df_out[COL_FROM_IDX] = g[COL_FROM_IDX].to_numpy()
        df_out[COL_TO_IDX] = g[COL_TO_IDX].to_numpy()

    changed = (COL_FROM_IDX not in df.columns) or (COL_TO_IDX not in df.columns)
    if changed:
        bak = pq.with_suffix(pq.suffix + ".bak_idx")
        if not bak.exists():
            shutil.copy2(pq, bak)
        df_out.to_parquet(pq, index=False)
        patched += 1
        print(f"[patched] {pq}")
    else:
        print(f"[ok] {pq} already has indices (overwriting not needed)")

print(f"Done. Patched {patched} parquet files.")
