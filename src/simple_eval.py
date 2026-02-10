import gymnasium as gym
import gym_pusht

import os
import numpy as np
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from lerobot.envs.factory import make_env
from lerobot.envs.configs import PushtEnv
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

DATASET = "pusht_cchi_v1_lerobotv3"

import imageio
from pathlib import Path

SAVE_MEDIA = True
SAVE_GIF = False
SAVE_MP4 = True
FPS = 10

OUT_DIR = Path(f"outputs/eval/lerobot/{DATASET}/pusht_videos")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RECORD_EPISODES = {0, 1, 2}
MAX_FRAMES = 500 


def transform_env_obs_to_policy_obs(env_obs: dict, device: torch.device) -> dict:
    """
    env obs keys: pixels (N,H,W,C) uint8, agent_pos (N,2) float
    policy expects: observation.images.img (N,C,H,W) float32 [0,1]
                    observation.state (N,5) float32
    Returns dict of torch.Tensors on device.
    """
    # pixels -> float32 [0,1], NHWC -> NCHW
    pix = np.asarray(env_obs["pixels"])
    if pix.dtype == np.uint8:
        pix = pix.astype(np.float32) / 255.0
    pix = np.transpose(pix, (0, 3, 1, 2))  # NHWC -> NCHW

    # agent_pos -> (N,5) pad zeros
    ap = np.asarray(env_obs["agent_pos"], dtype=np.float32)  # (N,2)
    N = ap.shape[0]
    state = np.zeros((N, 5), dtype=np.float32)
    state[:, :2] = ap

    obs = {
        "observation.images.img": torch.from_numpy(pix).to(device=device, dtype=torch.float32),
        "observation.state": torch.from_numpy(state).to(device=device, dtype=torch.float32),
    }
    return obs


def to_env_action(action) -> np.ndarray:
    """
    Convert policy output to numpy action for env.step().
    Handles torch tensors, numpy arrays, lists.
    """
    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().numpy()
    action = np.asarray(action)

    return action.astype(np.float32)


def main():
    ckpt_dir = f"outputs/train/lerobot/{DATASET}_forge/diffusion_pusht/checkpoints/last/pretrained_model"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_map = make_env(PushtEnv())
    tg = next(iter(env_map))
    tid = next(iter(env_map[tg]))
    env = env_map[tg][tid]

    policy = DiffusionPolicy.from_pretrained(ckpt_dir)
    policy.to(device).eval()

    n_episodes = 100
    returns = []
    successes = []

    for ep in range(n_episodes):
        out = env.reset()
        if isinstance(out, tuple) and len(out) == 2:
            env_obs, info = out
        else:
            env_obs, info = out, {}

        ep_ret = 0.0
        done = False

        print_and_save = False
        if (ep + 1) % 10 == 0:
            print_and_save = True

        frames = []
        if SAVE_MEDIA and print_and_save:
            frame0 = np.asarray(env_obs["pixels"][0])  # (H,W,C) uint8
            frames.append(frame0)

        while not done:
            obs = transform_env_obs_to_policy_obs(env_obs, device=device)

            with torch.no_grad():
                action = policy.select_action(obs)

            env_action = to_env_action(action)
            step_out = env.step(env_action)

            env_obs, reward, terminated, truncated, info = step_out

            if SAVE_MEDIA and print_and_save and len(frames) < MAX_FRAMES:
                frame = np.asarray(env_obs["pixels"][0])  # (H,W,C)
                frames.append(frame)

            r = float(np.asarray(reward).mean())
            ep_ret += r

            done = bool(np.asarray(terminated).any() or np.asarray(truncated).any())

        returns.append(ep_ret)

        success = None
        if isinstance(info, (list, tuple)) and len(info) > 0 and isinstance(info[0], dict):
            cand = info[0]
        elif isinstance(info, dict):
            cand = info
        else:
            cand = {}

        for k in ["success", "is_success", "task_success", "episode_success"]:
            if k in cand:
                success = bool(cand[k])
                break

        if success is not None:
            successes.append(success)

        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}/{n_episodes} return={ep_ret:.3f}")
        
        if SAVE_MEDIA and print_and_save and len(frames) > 0:
            if SAVE_GIF:
                gif_path = OUT_DIR / f"ep_{ep:04d}.gif"
                imageio.mimsave(gif_path, frames, fps=FPS)
                print("Saved GIF:", gif_path)

            if SAVE_MP4:
                mp4_path = OUT_DIR / f"ep_{ep:04d}.mp4"
                with imageio.get_writer(mp4_path, fps=FPS, codec="libx264", quality=8) as w:
                    for fr in frames:
                        w.append_data(fr)
                print("Saved MP4:", mp4_path)

    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns))
    print("\n=== Eval summary ===")
    print(f"episodes: {n_episodes}")
    print(f"return: mean={mean_ret:.3f}, std={std_ret:.3f}")

    if len(successes) > 0:
        print(f"success_rate: {100.0 * np.mean(successes):.1f}% (based on {len(successes)} episodes with a success flag)")
    else:
        print("success_rate: (env did not provide a success flag in info)")

if __name__ == "__main__":
    main()
