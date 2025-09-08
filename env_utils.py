import os
import numpy as np
from omegaconf import OmegaConf
import torch
import hydra
import sys
import gym
import gymnasium
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnvWrapper
import json

from dppo.env.gym_utils.wrapper import wrapper_dict
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils


def make_robomimic_env(render=False, env='square', normalization_path=None, low_dim_keys=None, dppo_path=None):
	wrappers = OmegaConf.create({
		'robomimic_lowdim': {
			'normalization_path': normalization_path,
			'low_dim_keys': low_dim_keys,
		},
	})
	obs_modality_dict = {
		"low_dim": (
			wrappers.robomimic_image.low_dim_keys
			if "robomimic_image" in wrappers
			else wrappers.robomimic_lowdim.low_dim_keys
		),
		"rgb": (
			wrappers.robomimic_image.image_keys
			if "robomimic_image" in wrappers
			else None
		),
	}
	if obs_modality_dict["rgb"] is None:
		obs_modality_dict.pop("rgb")
	ObsUtils.initialize_obs_modality_mapping_from_dict(obs_modality_dict)
	robomimic_env_cfg_path = f'{dppo_path}/cfg/robomimic/env_meta/{env}.json'
	with open(robomimic_env_cfg_path, "r") as f:
		env_meta = json.load(f)
	env_meta["reward_shaping"] = False
	env = EnvUtils.create_env_from_metadata(
		env_meta=env_meta,
		render=False,
		render_offscreen=render,
		use_image_obs=False,
	)
	env.env.hard_reset = False
	for wrapper, args in wrappers.items():
		env = wrapper_dict[wrapper](env, **args)
	return env


class ObservationWrapperRobomimic(gym.Env):
	def __init__(
		self,
		env,
		reward_offset=1,
	):
		self.env = env
		self.action_space = env.action_space
		self.observation_space = env.observation_space
		self.reward_offset = reward_offset

	def seed(self, seed=None):
		if seed is not None:
			np.random.seed(seed=seed)
		else:
			np.random.seed()

	def reset(self, **kwargs):
		options = kwargs.get("options", {})
		new_seed = options.get("seed", None)
		if new_seed is not None:
			self.seed(seed=new_seed)
		raw_obs = self.env.reset()
		obs = raw_obs['state'].flatten()
		return obs

	def step(self, action):
		raw_obs, reward, done, info = self.env.step(action)
		reward = (reward - self.reward_offset)
		obs = raw_obs['state'].flatten()
		return obs, reward, done, info

	def render(self, **kwargs):
		return self.env.render()
	

class ObservationWrapperGym(gym.Env):
	def __init__(
		self,
		env,
		normalization_path,
	):
		self.env = env
		self.action_space = env.action_space
		self.observation_space = env.observation_space
		normalization = np.load(normalization_path)
		self.obs_min = normalization["obs_min"]
		self.obs_max = normalization["obs_max"]
		self.action_min = normalization["action_min"]
		self.action_max = normalization["action_max"]

	def seed(self, seed=None):
		if seed is not None:
			np.random.seed(seed=seed)
		else:
			np.random.seed()

	def reset(self, **kwargs):
		options = kwargs.get("options", {})
		new_seed = options.get("seed", None)
		if new_seed is not None:
			self.seed(seed=new_seed)
		raw_obs = self.env.reset()
		obs = self.normalize_obs(raw_obs)
		return obs

	def step(self, action):
		raw_action = self.unnormalize_action(action)
		raw_obs, reward, done, info = self.env.step(raw_action)
		obs = self.normalize_obs(raw_obs)
		return obs, reward, done, info

	def render(self, **kwargs):
		return self.env.render()
	
	def normalize_obs(self, obs):
		return 2 * ((obs - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 0.5)

	def unnormalize_action(self, action):
		action = (action + 1) / 2
		return action * (self.action_max - self.action_min) + self.action_min
	

class ActionChunkWrapper(gymnasium.Env):
	def __init__(self, env, cfg, max_episode_steps=300):
		self.max_episode_steps = max_episode_steps
		self.env = env
		self.act_steps = cfg.act_steps
		self.action_space = spaces.Box(
			low=np.tile(env.action_space.low, cfg.act_steps),
			high=np.tile(env.action_space.high, cfg.act_steps),
			dtype=np.float32
		)
		self.observation_space = spaces.Box(
			low=-np.ones(cfg.obs_dim),
			high=np.ones(cfg.obs_dim),
			dtype=np.float32
		)
		self.count = 0

	def reset(self, seed=None):
		obs = self.env.reset(seed=seed)
		self.count = 0
		return obs, {}
	
	def step(self, action):
		if len(action.shape) == 1:
			action = action.reshape(self.act_steps, -1)
		obs_ = []
		reward_ = []
		done_ = []
		info_ = []
		done_i = False
		for i in range(action.shape[0]):
			self.count += 1
			obs_i, reward_i, done_i, info_i = self.env.step(action[i])
			obs_.append(obs_i)
			reward_.append(reward_i)
			done_.append(done_i)
			info_.append(info_i)
		obs = obs_[-1]
		reward = sum(reward_)
		done = np.max(done_)
		info = info_[-1]
		if self.count >= self.max_episode_steps:
			done = True
		if done:
			info['terminal_observation'] = obs
		return obs, reward, done, False, info

	def render(self):
		return self.env.render()
	
	def close(self):
		return
	

class DiffusionPolicyEnvWrapper(VecEnvWrapper):
	def __init__(self, env, cfg, base_policy):
		super().__init__(env)
		self.action_horizon = cfg.act_steps
		self.action_dim = cfg.action_dim
		self.action_space = spaces.Box(
			low=-cfg.train.action_magnitude*np.ones(self.action_dim*self.action_horizon),
			high=cfg.train.action_magnitude*np.ones(self.action_dim*self.action_horizon),
			dtype=np.float32
		)
		self.obs_dim = cfg.obs_dim
		self.observation_space = spaces.Box(
			low=-np.ones(self.obs_dim),
			high=np.ones(self.obs_dim),
			dtype=np.float32
		)
		self.env = env
		self.device = cfg.model.device
		self.base_policy = base_policy
		self.obs = None
		#residual getter
		self.residual_getter = None
		
	def set_residual_getter(self, fn):

		self.residual_getter = fn

	def step_async(self, actions):
		actions = torch.tensor(actions, device=self.device, dtype=torch.float32)
		B = actions.shape[0] #num_env
		H, A = self.action_horizon, self.action_dim
		expect_ha = H * A

		if actions.shape == (B, H, A):
			noise = actions
		elif actions.shape == (B, H * A):
			noise = actions.view(B, H, A)
		else:
			raise ValueError(
                f"Unexpected action shape {tuple(actions.shape)}; "
                f"expect {(B, H*A)} or {(B, H, A)}"
            )
		# DP(noise) -> norm
		norm_action = self.base_policy(self.obs, noise) # num_env * H * A
		if self.residual_getter is not None:
			# print("using residual")
			with torch.no_grad():
				res = self.residual_getter(self.obs)  # torch[B,H,A]
                # 保守起见，确保 shape 匹配
				if res.shape != norm_action.shape:
					raise RuntimeError(
                        f"residual_getter returned {tuple(res.shape)}, "
                        f"but expected {tuple(norm_action.shape)}"
                    )
		else:
			res = torch.zeros_like(noise)
		# print(f"norm_action shape:{norm_action.shape},norm_action dtype:{norm_action.dtype} ,res shape:{res.shape} ")
		# import pdb; pdb.set_trace()
		if isinstance(res, torch.Tensor):
			res = res.detach().cpu().numpy()
		else:
			res = np.asarray(res)
		res = res.astype(np.float32, copy=False)

		actual_norm_action = norm_action + res #res_coef TBD

		#store for add buffer
		self._curr_norm_action = norm_action
		self._curr_actual_norm_action = actual_norm_action
		self.venv.step_async(actual_norm_action)


	def step_wait(self):
		obs, rewards, dones, infos = self.venv.step_wait()
		if hasattr(self, "_curr_norm_action"):
			for i in range(len(infos)):
				infos[i]["norm_action"] = self._curr_norm_action[i]
				infos[i]["actual_norm_action"] = self._curr_actual_norm_action[i]
			del self._curr_norm_action
			del self._curr_actual_norm_action
		self.obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
		obs_out = self.obs
		return obs_out.detach().cpu().numpy(), rewards, dones, infos

	def reset(self):
		obs = self.venv.reset()
		self.obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
		obs_out = self.obs
		return obs_out.detach().cpu().numpy()
	
