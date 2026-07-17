import torch
import numpy as np
import copy
from typing import List, Optional

import genesis as gs
from genesis.utils.geom import (
    quat_to_xyz,
    transform_by_quat,
    inv_quat,
    transform_quat_by_quat,
    xyz_to_quat,
    quat_to_R,
)

from src.controller import build_controller
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv


class RaceEnv(VecEnv):
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        clip_a = env_cfg["clip_actions"]
        obs_high = np.concatenate([
            np.ones(3, dtype=np.float32),           # rel_pos (clipped)
            np.ones(4, dtype=np.float32),           # quat
            np.ones(3, dtype=np.float32),           # lin_vel (clipped)
            np.ones(3, dtype=np.float32),           # ang_vel (clipped)
            np.full(4, clip_a, dtype=np.float32),   # last_actions
        ])
        observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

        self.render_mode = None
        super().__init__(num_envs, observation_space, action_space)

        self.num_envs = num_envs
        self.rendered_env_num = min(10, self.num_envs)
        self.num_actions = env_cfg["num_actions"]
        self.cfg = env_cfg
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.dt = 0.01  # 100 Hz
        self.max_episode_length = np.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = copy.deepcopy(reward_cfg["reward_scales"])

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=5),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=env_cfg["max_visualize_FPS"],
                camera_pos=(3.0, 0.0, 3.0),
                camera_lookat=(0.0, 0.0, 1.0),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(
                show_world_frame=True,
                world_frame_size=0.5,
                show_link_frame=True,
                lights=[gs.options.vis.DirectionalLight(dir=(0.2, 0.4, -1), color=(1.0, 1.0, 1.0), intensity=5.0)],
                rendered_envs_idx=list(range(self.rendered_env_num)),
            ),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        # add plane
        self.scene.add_entity(gs.morphs.Plane())

        # add gates
        if self.env_cfg["visualize_target"]:
            for gate_pos, gate_rpy in zip(command_cfg["gates_position"], command_cfg["gates_rpy"]):
                self.scene.add_entity(
                    morph=gs.morphs.Mesh(
                        file="misc/gate.obj",
                        euler=(gate_rpy[0], gate_rpy[1], gate_rpy[2]),
                        pos=tuple(gate_pos),
                        fixed=True,
                        collision=False,
                    ),
                    surface=gs.surfaces.Rough(
                        diffuse_texture=gs.textures.ColorTexture(color=(0.0, 0.5, 0.5)),
                    ),
                )
        else:
            self.target = None

        # add camera
        if self.env_cfg["visualize_camera"]:
            self.cam = self.scene.add_camera(
                res=(640, 480),
                pos=(3.5, 0.0, 2.5),
                lookat=(0, 0, 0.5),
                fov=30,
                GUI=True,
            )

        # add drone
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=gs.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.drone = self.scene.add_entity(gs.morphs.Drone(
            file="misc/urdf/a300.urdf",
            propellers_spin=(-1, -1, 1, 1),
        ))

        # build scene
        self.scene.build(n_envs=num_envs)
        self.controller = build_controller(
            self.env_cfg["controller_type"], drone=self.drone, num_envs=self.num_envs, dt=self.dt, cfg=self.env_cfg
        )

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        # initialize buffers
        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_bool)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=gs.device, dtype=gs.tc_float)

        # racing track gates
        self.gates_position = torch.tensor(command_cfg["gates_position"], device=gs.device, dtype=gs.tc_float)
        self.gates_rpy = torch.tensor(command_cfg["gates_rpy"], device=gs.device, dtype=gs.tc_float)
        self.gates_R = quat_to_R(xyz_to_quat(self.gates_rpy, rpy=True, degrees=True))
        self.num_gates = self.gates_position.shape[0]
        self.gate_idx = torch.zeros((self.num_envs,), device=gs.device, dtype=torch.long)

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)

        self.base_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)

        self.rel_pos = torch.zeros_like(self.base_pos)
        self.last_rel_pos = torch.zeros_like(self.base_pos)
        self.last_base_pos = torch.zeros_like(self.base_pos)

        # SB3 episode statistics
        self.episode_returns = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.elapsed_time = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        self._pending_actions: Optional[torch.Tensor] = None

        self.reset()

    # ------------ SB3 VecEnv interface ----------------
    def reset(self) -> np.ndarray:
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        self._update_observation()
        return self._obs_np()

    def step_async(self, actions: np.ndarray) -> None:
        self._pending_actions = torch.as_tensor(actions, dtype=gs.tc_float, device=gs.device)

    @torch.inference_mode()
    def step_wait(self):
        torch.clamp(self._pending_actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"], out=self.actions)

        self.drone.set_propellers_rpm(self.controller.update(self.actions))
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.elapsed_time += self.dt
        self.last_base_pos[:] = self.base_pos[:]
        self.base_pos[:] = self.drone.get_pos()
        self.base_quat[:] = self.drone.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(self.inv_base_init_quat, self.base_quat), rpy=True, degrees=True
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.drone.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.drone.get_ang(), inv_base_quat)

        # gate crossing
        self.gate_success, self.gate_crash = self._at_gate()
        torch.where(self.gate_success, (self.gate_idx + 1) % self.num_gates, self.gate_idx, out=self.gate_idx)
        self.commands.copy_(self.gates_position[self.gate_idx])
        torch.sub(self.commands, self.base_pos, out=self.rel_pos)
        torch.sub(self.commands, self.last_base_pos, out=self.last_rel_pos)

        # terminations
        self.termination_conditions = (
            (torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"])
            | (torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"])
            | (torch.abs(self.base_pos[:, 0]) > self.env_cfg["arena_half_x"])
            | (torch.abs(self.base_pos[:, 1]) > self.env_cfg["arena_half_y"])
            | (self.base_pos[:, 2] > self.env_cfg["arena_z_max"])
            | (self.base_pos[:, 2] < self.env_cfg["termination_if_close_to_ground"])
            | self.gate_crash
            | self.scene.rigid_solver.get_error_envs_mask()
        )
        timed_out = self.episode_length_buf > self.max_episode_length
        dones = timed_out | self.termination_conditions

        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        self.episode_returns += self.rew_buf

        self._update_observation()
        obs = self._obs_np()
        rew_np = self.rew_buf.cpu().numpy().astype(np.float32)
        done_np = dones.cpu().numpy()

        self.last_actions[:] = self.actions[:]

        infos: List[dict] = [{} for _ in range(self.num_envs)]
        if dones.any():
            done_idx = dones.nonzero(as_tuple=True)[0]
            timed_out_np = timed_out.cpu().numpy()
            terminated_np = self.termination_conditions.cpu().numpy()

            for i in done_idx.tolist():
                infos[i]["terminal_observation"] = obs[i].copy()
                if timed_out_np[i] and not terminated_np[i]:
                    infos[i]["TimeLimit.truncated"] = True
                infos[i]["episode"] = {
                    "r": self.episode_returns[i].item(),
                    "l": int(self.episode_length_buf[i].item()),
                    "t": self.elapsed_time[i].item(),
                }

            self.episode_returns[done_idx] = 0.0
            self.elapsed_time[done_idx] = 0.0
            self.reset_idx(done_idx)

            # replace obs of reset envs with fresh post-reset obs
            self._update_observation()
            new_obs = self._obs_np()
            done_idx_np = done_idx.cpu().numpy()
            obs[done_idx_np] = new_obs[done_idx_np]

        return obs, rew_np, done_np, infos

    def close(self) -> None:
        pass

    def get_attr(self, attr_name, indices=None):
        n = len(self._get_indices(indices))
        return [getattr(self, attr_name)] * n

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        n = len(self._get_indices(indices))
        result = getattr(self, method_name)(*method_args, **method_kwargs)
        return [result] * n

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(self._get_indices(indices))

    def seed(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        return [seed] * self.num_envs

    # ------------ internals ----------------
    def _resample_commands(self, envs_idx):
        self.commands[envs_idx] = self.gates_position[self.gate_idx[envs_idx]]

    def _at_gate(self):
        R = self.gates_R[self.gate_idx]
        c = self.gates_position[self.gate_idx]
        Rt = R.transpose(1, 2)

        prev = torch.einsum("nij,nj->ni", Rt, self.last_base_pos - c)
        curr = torch.einsum("nij,nj->ni", Rt, self.base_pos - c)

        plane_crossed = (prev[:, 1] < 0.0) & (curr[:, 1] >= 0.0)
        inside_gate = (curr[:, 0].abs() < self.env_cfg["gate_half_width"]) & (
            curr[:, 2].abs() < self.env_cfg["gate_half_height"]
        )

        return plane_crossed & inside_gate, plane_crossed & ~inside_gate

    def _update_observation(self):
        self.obs_buf = torch.cat(
            [
                torch.clip(self.rel_pos * self.obs_scales["rel_pos"], -1, 1),
                self.base_quat,
                torch.clip(self.base_lin_vel * self.obs_scales["lin_vel"], -1, 1),
                torch.clip(self.base_ang_vel * self.obs_scales["ang_vel"], -1, 1),
                self.last_actions,
            ],
            axis=-1,
        )

    def _obs_np(self) -> np.ndarray:
        return self.obs_buf.cpu().numpy().astype(np.float32)

    # ------------ reset ----------------
    def _sample_spawn(self, envs_idx):
        n = len(envs_idx)
        R = self.gates_R[self.gate_idx[envs_idx]]
        c = self.gates_position[self.gate_idx[envs_idx]]

        def u(lo, hi):
            return torch.empty((n,), device=gs.device, dtype=gs.tc_float).uniform_(lo, hi)

        lo, hi = self.env_cfg["spawn_back_dist"]
        local = torch.stack(
            [
                u(-self.env_cfg["spawn_lateral"], self.env_cfg["spawn_lateral"]),
                -u(lo, hi),
                u(-self.env_cfg["spawn_vertical"], self.env_cfg["spawn_vertical"]),
            ],
            dim=-1,
        )
        pos = c + torch.einsum("nij,nj->ni", R, local)
        pos[:, 2] = pos[:, 2].clamp(min=self.env_cfg["spawn_min_height"])

        fwd = R[:, :, 1]
        gate_yaw = torch.atan2(fwd[:, 1], fwd[:, 0])
        yaw = gate_yaw - np.pi / 2 + u(-np.pi / 6, np.pi / 6)
        yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))

        rpy = torch.stack([torch.zeros_like(yaw), torch.zeros_like(yaw), yaw], dim=-1)
        quat = xyz_to_quat(rpy, rpy=True, degrees=False)

        return pos, quat

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        self.gate_idx[envs_idx] = torch.randint(0, self.num_gates, (len(envs_idx),), device=gs.device)
        pos, quat = self._sample_spawn(envs_idx)

        self.base_pos[envs_idx] = pos
        self.last_base_pos[envs_idx] = pos
        self.base_quat[envs_idx] = quat
        self.drone.set_pos(self.base_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.drone.set_quat(self.base_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.drone.zero_all_dofs_velocity(envs_idx)

        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True
        self.controller.reset_idx(envs_idx)

        for key in self.episode_sums.keys():
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)
        torch.sub(self.commands, self.base_pos, out=self.rel_pos)
        torch.sub(self.commands, self.last_base_pos, out=self.last_rel_pos)

    # ------------ reward functions ----------------
    def _reward_progress(self):
        progress_rew = torch.sum(torch.square(self.last_rel_pos), dim=1) - torch.sum(torch.square(self.rel_pos), dim=1)
        return torch.where(self.gate_success, torch.zeros_like(progress_rew), progress_rew)

    def _reward_smooth(self):
        return torch.sum(torch.square(self.actions - self.last_actions), dim=1)

    def _reward_yaw(self):
        yaw = self.base_euler[:, 2]
        yaw = torch.where(yaw > 180, yaw - 360, yaw) / 180 * 3.14159
        return torch.exp(self.reward_cfg["yaw_lambda"] * torch.abs(yaw))

    def _reward_angular(self):
        return torch.norm(self.base_ang_vel / 3.14159, dim=1)

    def _reward_crash(self):
        crash_rew = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        crash_rew[self.termination_conditions] = 1
        return crash_rew