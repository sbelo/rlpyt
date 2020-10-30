from gym import Wrapper
from gym.wrappers.time_limit import TimeLimit
from rlpyt.envs.base import EnvSpaces, EnvStep
from rlpyt.spaces.gym_wrapper import GymSpaceWrapper
from rlpyt.spaces.int_box import IntBox
from gym.spaces.box import Box
import random
import math
import numpy as np
from rlpyt.envs.atari.atari_env import AtariEnv, AtariTrajInfo

def obs_action_translator(action,window_size,dim_obs):
    trans_action = np.zeros(dim_obs)
    x_inds = np.arange(dim_obs[1])
    y_inds = np.arange(dim_obs[0])
    x_inds = x_inds[int(max(math.floor(action[0] - window_size[0]/2),0)):min(math.ceil(action[0] + window_size[0] / 2),dim_obs[0])]
    y_inds = y_inds[int(max(math.floor(action[1] - window_size[1]/2),0)):min(math.ceil(action[1] + window_size[1] / 2),dim_obs[1])]
    trans_action[:,x_inds,y_inds] = 1
    return trans_action

def reward_shaping_ph(reward):
    return reward, 0


class CWTO_EnvWrapperAtari(Wrapper):
    def __init__(self,env_name,window_size,force_float32=True,player_reward_shaping=None,observer_reward_shaping=None,max_episode_length=np.inf,add_channel=False):
        self.serial = False
        env = AtariEnv(game=env_name)
        env.metadata = None
        env.reward_range = None
        super().__init__(env)
        o = self.env.reset()
        self.max_episode_length = max_episode_length
        self.curr_episode_length = 0
        self.add_channel = add_channel
        o, r, d, info = self.env.step(self.env.action_space.sample())
        env_ = self.env
        time_limit = isinstance(self.env, TimeLimit)
        while not time_limit and hasattr(env_, "env"):
            env_ = env_.env
            time_limit = isinstance(env_, TimeLimit)
        if time_limit:
            info["timeout"] = False  # gym's TimeLimit.truncated invalid name.
        self.time_limit = time_limit
        self._action_space = GymSpaceWrapper(
            space=self.env.action_space,
            name="act",
            null_value=self.env.action_space.null_value(),
            force_float32=force_float32,
        )
        self._observation_space = GymSpaceWrapper(
            space=self.env.observation_space,
            name="obs",
            null_value=self.env.observation_space.null_value(),
            force_float32=force_float32,
        )
        del self.action_space
        del self.observation_space
        self.player_turn = False
        self.last_done = False
        self.last_reward = 0
        self.last_info = {}
        if player_reward_shaping is None:
            self.player_reward_shaping = reward_shaping_ph
        else:
            self.player_reward_shaping = player_reward_shaping
        if observer_reward_shaping is None:
            self.observer_reward_shaping = reward_shaping_ph
        else:
            self.observer_reward_shaping = observer_reward_shaping
        self.obs_size = self.env.observation_space.shape
        self.window_size = window_size
        self.obs_action_translator = obs_action_translator

        player_obs_space = self.env.observation_space
        if add_channel:
            player_obs_space = IntBox(low=player_obs_space.low,high=player_obs_space.high,shape=player_obs_space.shape,dtype=player_obs_space.dtype,null_value=player_obs_space.null_value())
        player_act_space = self.env.action_space
        observer_obs_space = self.env.observation_space
        observer_act_space = Box(low=np.asarray([0.0,0.0]),high=np.asarray([self.env.observation_space.shape[0],self.env.observation_space.shape[1]]))

        self.player_action_space = GymSpaceWrapper(space=player_act_space,name="act",null_value=player_act_space.null_value(),force_float32=force_float32)
        self.observer_action_space = GymSpaceWrapper(space=observer_act_space,name="act",null_value=np.zeros(2),force_float32=force_float32)
        self.player_observation_space = GymSpaceWrapper(space=player_obs_space,name="obs",null_value=player_obs_space.null_value,force_float32=force_float32)
        self.observer_observation_space = GymSpaceWrapper(space=observer_obs_space,name="obs",null_value=observer_obs_space.null_value(),force_float32=force_float32)

    def step(self,action):
        if self.player_turn:
            self.player_turn = False
            a = self.player_action_space.revert(action)
            if a.size <= 1:
                a = a.item()
            o, r, d, info = self.env.step(a)
            self.last_obs = o
            self.last_action = a
            obs = self.observer_observation_space.convert(o)
            if self.time_limit:
                if "TimeLimit.truncated" in info:
                    info["timeout"] = info.pop("TimeLimit.truncated")
                else:
                    info["timeout"] = False
            
            self.last_info = info #(info["timeout"])
#             info = (False)
            if isinstance(r, float):
                r = np.dtype("float32").type(r)  # Scalar float32.
            self.last_reward = r
            self.curr_episode_length += 1
            if self.curr_episode_length >= self.max_episode_length:
                d = True
            self.last_done = d
            return EnvStep(obs, r, d, info)

        else:
            r_action = self.observer_action_space.revert(action)
            r_action = self.obs_action_translator(r_action, self.window_size, self.obs_size)
            self.player_turn = True
            self.last_obs_act = r_action
            masked_obs = np.multiply(r_action, self.last_obs)
            info = self.last_info
            r = self.last_reward
            d = self.last_done
            if self.add_channel:
                masked_obs = np.concatenate([r_action,masked_obs],axis=0)
            else:
                masked_obs[r_action == 0] = -1
            obs = self.player_observation_space.convert(masked_obs)

            return EnvStep(obs,r,d,info)


    def reset(self):
        self.curr_episode_length = 0
        self.last_done = False
        self.last_reward = 0
        self.last_action = self.player_action_space.revert(self.player_action_space.null_value())
        self.player_turn = False
        o = self.env.reset()
        self.last_obs = o
        self.last_obs_act = np.zeros(o.shape)
        obs = self.observer_observation_space.convert(o)
        return obs

    def spaces(self):
        comb_spaces = [EnvSpaces(observation=self.player_observation_space,action=self.player_action_space), EnvSpaces(observation=self.observer_observation_space, action=self.observer_action_space)]
        return comb_spaces

    def action_space(self):
        if self.player_turn:
            return self.player_action_space
        else:
            return self.observer_action_space

    def observation_space(self):
        if self.player_turn:
            return self.player_observation_space
        else:
            return self.observer_observation_space

