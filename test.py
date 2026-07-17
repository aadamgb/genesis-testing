import argparse
import os
import pickle

import numpy as np
import torch

import genesis as gs
from stable_baselines3 import PPO

from src.env import RaceEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="fig8/sb3")
    parser.add_argument("-c", "--ckpt", type=str, default=None,
                        help="path to a .zip checkpoint; defaults to <log_dir>/<name>_final.zip")
    parser.add_argument("-B", "--num_envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--stochastic", action="store_true",
                        help="sample from the policy instead of using the mean action")
    parser.add_argument("--record", action="store_true",
                        help="record camera video instead of interactive viewer")
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", performance_mode=True)

    log_dir = f"logs/{args.exp_name}"
    with open(f"{log_dir}/cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg = pickle.load(f)

    # visualization overrides
    env_cfg["visualize_target"] = True
    env_cfg["visualize_camera"] = args.record
    env_cfg["max_visualize_FPS"] = 100  # match env dt (100 Hz) for real-time playback

    env = RaceEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=not args.record,
    )

    ckpt = args.ckpt or os.path.join(log_dir, f"{os.path.basename(args.exp_name)}_final.zip")
    model = PPO.load(ckpt, device="cpu")  # inference: tiny MLP, CPU is fine
    print(f"Loaded {ckpt}")

    obs = env.reset()
    if args.record:
        env.cam.start_recording()

    ep_returns, gates_passed = [], 0
    with torch.no_grad():
        for step in range(args.steps):
            actions, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, rewards, dones, infos = env.step(actions)

            gates_passed += int(env.gate_success.sum().item())
            for info in infos:
                if "episode" in info:
                    ep_returns.append(info["episode"]["r"])
                    print(f"episode done: r={info['episode']['r']:.2f}  "
                          f"l={info['episode']['l']}  t={info['episode']['t']:.2f}s")

            if args.record:
                env.cam.render()

    if args.record:
        env.cam.stop_recording(save_to_filename=f"{log_dir}/eval.mp4", fps=100)
        print(f"saved {log_dir}/eval.mp4")

    if ep_returns:
        print(f"\n{len(ep_returns)} episodes | mean return {np.mean(ep_returns):.2f} "
              f"± {np.std(ep_returns):.2f} | gate crossings: {gates_passed}")

    env.close()


if __name__ == "__main__":
    main()

"""
python test.py -e fig8/sb3                        # interactive viewer, mean policy
python test.py -e fig8/sb3 -c logs/fig8/sb3/sb3_2000000_steps.zip
python test.py -e fig8/sb3 --record               # offscreen -> eval.mp4
"""