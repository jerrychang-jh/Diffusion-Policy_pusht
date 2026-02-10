from pathlib import Path
import re, json, shutil, subprocess
import pandas as pd

ROOT = Path("Dataset/lerobot_v3/pusht_cchi_v1_lerobotv3")
VIDEO_KEY = "observation.images.img"

COLS = {
    "chunk_index": f"videos/{VIDEO_KEY}/chunk_index",
    "file_index":  f"videos/{VIDEO_KEY}/file_index",
    "from_ts":     f"videos/{VIDEO_KEY}/from_timestamp",
    "to_ts":       f"videos/{VIDEO_KEY}/to_timestamp",
    "fps":         f"videos/{VIDEO_KEY}/fps",
    "nframes":     f"videos/{VIDEO_KEY}/num_frames",
}

episodes_root = ROOT / "meta" / "episodes"
videos_root   = ROOT / "videos" / VIDEO_KEY

chunk_re = re.compile(r"chunk-(\d+)")
parquets = sorted(episodes_root.glob("chunk-*/**/*.parquet"))
if not parquets:
    raise RuntimeError(f"No parquet files found under {episodes_root}")

def ffprobe_info(mp4: Path):
    """
    Returns (duration_seconds, fps, num_frames_best_effort)
    """
    # Grab stream info as json
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,r_frame_rate,nb_frames,duration",
        "-of", "json",
        str(mp4)
    ]
    out = subprocess.check_output(cmd).decode("utf-8")
    info = json.loads(out)
    st = info["streams"][0]

    # duration
    duration = st.get("duration")
    duration = float(duration) if duration is not None else None

    # fps from avg_frame_rate
    def parse_frac(s):
        if not s or s == "0/0":
            return None
        a,b = s.split("/")
        return float(a) / float(b)

    fps = parse_frac(st.get("avg_frame_rate")) or parse_frac(st.get("r_frame_rate"))

    # frames
    nb = st.get("nb_frames")
    nframes = int(nb) if nb and nb.isdigit() else None

    # fallback if missing
    if (nframes is None) and (duration is not None) and (fps is not None):
        nframes = int(round(duration * fps))

    if duration is None and nframes is not None and fps is not None:
        duration = nframes / fps

    if duration is None or fps is None:
        raise RuntimeError(f"ffprobe missing duration/fps for {mp4}: {st}")

    return duration, fps, nframes

patched = 0
for pq in parquets:
    m = chunk_re.search(str(pq.parent))
    if not m:
        print(f"[skip] cannot infer chunk id from path: {pq}")
        continue
    chunk_idx = int(m.group(1))

    mp4 = videos_root / f"chunk-{chunk_idx:03d}" / "file-000.mp4"
    if not mp4.exists():
        raise FileNotFoundError(f"Missing mp4 for chunk {chunk_idx}: {mp4}")

    duration, fps, nframes = ffprobe_info(mp4)

    df = pd.read_parquet(pq)
    changed = False

    # required mapping fields
    if COLS["chunk_index"] not in df.columns:
        df[COLS["chunk_index"]] = chunk_idx
        changed = True
    if COLS["file_index"] not in df.columns:
        df[COLS["file_index"]] = 0
        changed = True

    # time fields (episode-local)
    if COLS["from_ts"] not in df.columns:
        df[COLS["from_ts"]] = 0.0
        changed = True
    if COLS["to_ts"] not in df.columns:
        df[COLS["to_ts"]] = float(duration)
        changed = True

    # fps / frames
    if COLS["fps"] not in df.columns:
        df[COLS["fps"]] = float(fps)
        changed = True
    if COLS["nframes"] not in df.columns:
        df[COLS["nframes"]] = int(nframes) if nframes is not None else int(round(duration * fps))
        changed = True

    if changed:
        bak = pq.with_suffix(pq.suffix + ".bak2")
        if not bak.exists():
            shutil.copy2(pq, bak)
        df.to_parquet(pq, index=False)
        patched += 1
        print(f"[patched] {pq} (chunk={chunk_idx}, dur={duration:.3f}s, fps={fps:.3f}, frames={nframes})")
    else:
        print(f"[ok] {pq} already has required columns")

print(f"Done. Patched {patched} parquet files.")
