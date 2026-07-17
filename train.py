import argparse
import os
import pickle
import shutil

import torch
import yaml

import genesis as gs
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from src.env import RaceEnv   # the SB3-refactored version


def load_track(track_path):
    with open(track_path, "r") as f:
        track = yaml.safe_load(f)
    return {
        "gates_pos": [g["position"] for g in track["gates"]],
        "gates_rpy": [g["rpy"] for g in track["gates"]],
        "limits": {k: v for d in track["limits"] for k, v in d.items()},
    }


def get_cfgs(cm):
    track = load_track("misc/fig8.yaml")

    env_cfg = {
        "num_actions": 4,

        # controller
        "controller_type": cm,
        "hover_rpm": 8120.65,
        "action_scale": 0.8,
        "ctbr_mixer": [
            [ 1,  1,  1,  1],
            [-1,  1,  1, -1],
            [-1,  1, -1,  1],
            [-1, -1,  1,  1],
        ],

        # termination
        "termination_if_roll_greater_than": 180,
        "termination_if_pitch_greater_than": 180,
        "termination_if_close_to_ground": 0.1,
        "arena_half_x": track["limits"]["arena_half_x"],
        "arena_half_y": track["limits"]["arena_half_y"],
        "arena_z_max": track["limits"]["arena_z_max"],

        # base pose
        "base_init_pos": [-1.5, -2.5, 1.0],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 12.0,
        "at_target_threshold": 0.1,
        "gate_half_width": 0.6,
        "gate_half_height": 0.6,
        "resampling_time_s": 3.0,
        "simulate_action_latency": True,
        "clip_actions": 1.0,

        # spawn
        "spawn_back_dist": [0.8, 1.2],
        "spawn_lateral": 0.4,
        "spawn_vertical": 0.3,
        "spawn_min_height": 0.5,

        # visualization
        "visualize_target": False,
        "visualize_camera": False,
        "max_visualize_FPS": 60,
    }

    obs_cfg = {
        "obs_scales": {
            "rel_pos": 1 / 3.0,
            "lin_vel": 1 / 10.0,
            "ang_vel": 1 / 3.14159,
        },
    }

    reward_cfg = {
        "yaw_lambda": -10.0,
        "reward_scales": {
            "progress": 10.0,
            "smooth": -1e-4,
            "yaw": 0.0,
            "angular": -2e-4,
            "crash": -10.0,
        },
    }

    command_cfg = {
        "num_commands": 3,
        "gates_position": track["gates_pos"],
        "gates_rpy": track["gates_rpy"],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="sb3_test")
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-B", "--num_envs", type=int, default=8192)
    parser.add_argument("-t", "--total_timesteps", type=int, default=100_000_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("-cm", "--control_mode", type=str, default="SRT")
    parser.add_argument("--n_steps", type=int, default=100)
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning",
            seed=args.seed, performance_mode=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs(args.control_mode)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    if args.vis:
        env_cfg["visualize_target"] = True

    # with open(f"{log_dir}/cfgs.pkl", "wb") as f:
    #     pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg], f)

    env = RaceEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.vis,
    )

    # rollout buffer = n_steps * num_envs; 4 minibatches per epoch (rsl-rl parity)
    rollout_size = args.n_steps * args.num_envs
    batch_size = rollout_size // 4

    policy_kwargs = dict(
        activation_fn=torch.nn.Tanh,
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
        log_std_init=0.0,   # exp(0) = 1.0 -> matches rsl-rl init_std=1.0
    )

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,
        seed=args.seed,
        n_steps=args.n_steps,
        batch_size=batch_size,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.004,
        vf_coef=1.0,
        max_grad_norm=1.0,
        learning_rate=3e-4,
        target_kl=None,
        tensorboard_log=log_dir,
    )

    # checkpoint_callback = CheckpointCallback(
    #     save_freq=max(1_000_000 // args.num_envs, 1),  # save_freq is per-env steps
    #     save_path=log_dir,
    #     name_prefix=args.exp_name,
    # )

    # model.learn(total_timesteps=args.total_timesteps, callback=checkpoint_callback)
    model.learn(total_timesteps=args.total_timesteps)
    model.save(f"logs/fig8/sb3/sb3_final")
    env.close()


if __name__ == "__main__":
    main()

"""
python train_sb3.py -e srt_run1 -cm SRT -B 1024
"""