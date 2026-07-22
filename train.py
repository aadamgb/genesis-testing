import argparse
import os
import pickle
import shutil
from importlib import metadata
import yaml
import hydra
from omegaconf import OmegaConf, DictConfig

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from src.env import RaceEnv


# def get_train_cfg(exp_name):
#     train_cfg_dict = {
#         "algorithm": {
#             "class_name": "PPO",
#             "clip_param": 0.2,
#             "desired_kl": 0.01,
#             "entropy_coef": 0.004,
#             "gamma": 0.99,
#             "lam": 0.95,
#             "learning_rate": 0.0003,
#             "max_grad_norm": 1.0,
#             "num_learning_epochs": 5,
#             "num_mini_batches": 4,
#             "schedule": "adaptive",
#             "use_clipped_value_loss": True,
#             "value_loss_coef": 1.0,
#         },
#         "actor": {
#             "class_name": "MLPModel",
#             "hidden_dims": [128, 128],
#             "activation": "tanh",
#             "distribution_cfg": {
#                 "class_name": "GaussianDistribution",
#                 "init_std": 1.0,
#                 "std_type": "scalar",
#             },
#         },
#         "critic": {
#             "class_name": "MLPModel",
#             "hidden_dims": [128, 128],
#             "activation": "tanh",
#         },
#         "obs_groups": {
#             "actor": ["policy"],
#             "critic": ["policy"],
#         },
#         "num_steps_per_env": 100,
#         "save_interval": 100,
#         "run_name": exp_name,
#         "logger": "tensorboard",
#     }

#     return train_cfg_dict


def load_track(track_path):
    with open(track_path, "r") as f:
        track = yaml.safe_load(f)
    return {
        "gates_pos": [g["position"] for g in track["gates"]],
        "gates_rpy": [g["rpy"] for g in track["gates"]],
        "limits": {k: v for d in track["limits"] for k, v in d.items()}
    }

# def get_cfgs(cm):

#     track = load_track("misc/fig8.yaml")

#     env_cfg = {
#         "num_actions": 4,

#         # controller
#         "controller_type": cm,  
#         # "hover_rpm": 15502.5,
#         "hover_rpm": 8120.65,
#         "action_scale": 0.8,
#         "ctbr_mixer": [             
#             [ 1,   1,   1,   1],
#             [-1,   1,   1,  -1],
#             [-1,   1,  -1,   1],
#             [-1,  -1,   1,   1],
#         ],

#         # termination
#         "termination_if_roll_greater_than": 180,  # degree
#         "termination_if_pitch_greater_than": 180,
#         "termination_if_close_to_ground": 0.1,
#         "arena_half_x": track["limits"]["arena_half_x"],
#         "arena_half_y": track["limits"]["arena_half_y"],
#         "arena_z_max": track["limits"]["arena_z_max"],

#         # base pose
#         "base_init_pos": [-1.5, -2.5, 1.0],
#         "base_init_quat": [1.0, 0.0, 0.0, 0.0],
#         "episode_length_s": 12.0,
#         "at_target_threshold": 0.1,
#         "gate_half_width": 0.6,
#         "gate_half_height": 0.6,
#         "resampling_time_s": 3.0,
#         "simulate_action_latency": True,
#         "clip_actions": 1.0,

#         # spawn
#         "spawn_back_dist": [0.8, 1.2],   # metres behind the gate plane
#         "spawn_lateral": 0.4,            # ± in gate x
#         "spawn_vertical": 0.3,           # ± in gate z
#         "spawn_min_height": 0.5,

#         # visualization
#         "visualize_target": False,
#         "visualize_camera": False,
#         "max_visualize_FPS": 60,
#     }

#     obs_cfg = {
#         "obs_scales": {
#             "rel_pos": 1 / 3.0,
#             "lin_vel": 1 / 3.0,
#             "ang_vel": 1 / 3.14159,
#         },
#     }

#     reward_cfg = {
#         "yaw_lambda": -10.0,
#         "reward_scales": {
#             "progress": 10.0,
#             "smooth": -1e-4,
#             "yaw": 0.0,
#             "angular": -2e-4,
#             "crash": -10.0,
#         },
#     }
    
#     command_cfg = {
#         "num_commands": 3,
#         "gates_position": track["gates_pos"],
#         "gates_rpy": track["gates_rpy"],
#     }

#     return env_cfg, obs_cfg, reward_cfg, command_cfg


@hydra.main(version_base=None, config_path="hydra_configs", config_name="train")
def main(cfg: DictConfig):
    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=cfg.seed, performance_mode=True)

    task = cfg.task
    env_cfg     = OmegaConf.to_container(task.env, resolve=True)
    obs_cfg     = OmegaConf.to_container(task.obs, resolve=True)
    reward_cfg  = OmegaConf.to_container(task.reward, resolve=True)
    command_cfg = OmegaConf.to_container(task.command, resolve=True)
    train_cfg   = OmegaConf.to_container(task.rsl_rl, resolve=True)

    if task.get("track_path"):
        track = load_track(task.track_path)
        env_cfg.update(track["limits"])
        command_cfg["gates_position"] = track["gates_pos"]
        command_cfg["gates_rpy"]      = track["gates_rpy"]

    log_dir = task.log_dir
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    with open(f"{log_dir}/cfgs.pkl", "wb") as f:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)

    if task.name == "racing":
        env = RaceEnv(num_envs=task.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg,
                      reward_cfg=reward_cfg, command_cfg=command_cfg, show_viewer=task.vis)
    # elif task.name == "sprind":
    #     from src.sprind_env import SprindEnv
    #     env = SprindEnv(num_envs=task.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg,
    #                     reward_cfg=reward_cfg, command_cfg=command_cfg, show_viewer=task.vis)
    else:
        raise ValueError(f"unknown env: {task.name}")

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    runner.learn(num_learning_iterations=cfg.m, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# training
python examples/drone/hover_train.py
"""
