import numpy as np
from gym import Wrapper
from gym.wrappers.time_limit import TimeLimit
from rlpyt.envs.base import EnvSpaces, EnvStep
from rlpyt.spaces.gym_wrapper import GymSpaceWrapper
import random
import numpy as np


def obs_action_translator(action,power_vec,dim_obs):
    trans_action = np.zeros(dim_obs)
    for i in range(len(power_vec)):
        if action >= power_vec[i]:
            trans_action[i] = 1
            action -= power_vec[i]
    return trans_action

def reward_shaping_ph(reward,obs_act):
    return reward, 0


class CWTO_EnvWrapper(Wrapper):
    def __init__(self,work_env,env_name,obs_spaces,action_spaces,serial,force_float32=True,act_null_value=[0,0],obs_null_value=[0,0],player_reward_shaping=None,observer_reward_shaping=None,fully_obs=False,rand_obs=False,inc_player_last_act=False,max_episode_length=np.inf,cont_act=False):
        env = work_env(env_name)
        super().__init__(env)
        o = self.env.reset()
        self.inc_player_last_act = inc_player_last_act
        self.max_episode_length = max_episode_length
        self.curr_episode_length = 0
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
            null_value=act_null_value,
            force_float32=force_float32,
        )
        self.cont_act = cont_act
        self._observation_space = GymSpaceWrapper(
            space=self.env.observation_space,
            name="obs",
            null_value=obs_null_value,
            force_float32=force_float32,
        )
        del self.action_space
        del self.observation_space
        self.fully_obs = fully_obs
        self.rand_obs = rand_obs
        self.player_turn = False
        self.serial = serial
        self.last_done = False
        self.last_reward = 0
        self.last_info = {}
        # self.obs_action_translator = obs_action_translator
        if player_reward_shaping is None:
            self.player_reward_shaping = reward_shaping_ph
        else:
            self.player_reward_shaping = player_reward_shaping
        if observer_reward_shaping is None:
            self.observer_reward_shaping = reward_shaping_ph
        else:
            self.observer_reward_shaping = observer_reward_shaping
        dd = self.env.observation_space.shape
        obs_size = 1
        for d in dd:
            obs_size *= d
        self.obs_size = obs_size
        if serial:
            self.ser_cum_act = np.zeros(self.env.observation_space.shape)
            self.ser_counter = 0
        else:
            self.power_vec = 2 ** np.arange(self.obs_size)[::-1]
            self.obs_action_translator = obs_action_translator
        if len(obs_spaces) > 1:
            player_obs_space = obs_spaces[0]
            observer_obs_space = obs_spaces[1]
        else:
            player_obs_space = obs_spaces[0]
            observer_obs_space = obs_spaces[0]
        if len(action_spaces) > 1:
            player_act_space = action_spaces[0]
            observer_act_space = action_spaces[1]
        else:
            player_act_space = action_spaces[0]
            observer_act_space = action_spaces[0]


        self.player_action_space = GymSpaceWrapper(space=player_act_space,name="act",null_value=act_null_value[0],force_float32=force_float32)
        self.observer_action_space = GymSpaceWrapper(space=observer_act_space,name="act",null_value=act_null_value[1],force_float32=force_float32)
        self.player_observation_space = GymSpaceWrapper(space=player_obs_space,name="obs",null_value=obs_null_value[0],force_float32=force_float32)
        self.observer_observation_space = GymSpaceWrapper(space=observer_obs_space,name="obs",null_value=obs_null_value[1],force_float32=force_float32)

    def step(self,action):
        if self.player_turn:
            self.player_turn = False
            a = self.player_action_space.revert(action)
            if a.size <= 1:
                a = a.item()
            o, r, d, info = self.env.step(a)
            self.last_obs = o
            self.last_action = a
            if self.serial:
                obs = np.concatenate([np.zeros(self.last_obs_act.shape), self.last_masked_obs])
            else:
                obs = np.concatenate([self.last_obs_act, self.last_masked_obs])
            if self.inc_player_last_act:
                obs = np.append(obs,a)
            obs = self.observer_observation_space.convert(obs)
            if self.time_limit:
                if "TimeLimit.truncated" in info:
                    info["timeout"] = info.pop("TimeLimit.truncated")
                else:
                    info["timeout"] = False

            self.last_info = (info["timeout"])
            info = (False)
            if isinstance(r, float):
                r = np.dtype("float32").type(r)  # Scalar float32.
            self.last_reward = r
            # if (not d) and (self.observer_reward_shaping is not None):
            #     r = self.observer_reward_shaping(r,self.last_obs_act)
            self.curr_episode_length += 1
            if self.curr_episode_length >= self.max_episode_length:
                d = True
            self.last_done = d
            return EnvStep(obs, r, d, info)

        else:
            if not np.array_equal(action,action.astype(bool)):
                action = np.random.binomial(1,action)
            r_action = self.observer_action_space.revert(action)
            if self.serial:
                if self.fully_obs:
                    r_action = 1
                elif self.rand_obs:
                    r_action = random.randint(0,1)
                self.ser_cum_act[self.ser_counter] = r_action
                self.ser_counter += 1
                if self.ser_counter == self.obs_size:
                    self.player_turn = True
                    self.ser_counter = 0
                    masked_obs = np.multiply(np.reshape(self.ser_cum_act, self.last_obs.shape), self.last_obs)
                    self.last_masked_obs = masked_obs
                    self.last_obs_act = self.ser_cum_act.copy()
                    self.ser_cum_act = np.zeros(self.env.env.observation_space.shape)
                    r = self.last_reward
                    # if self.player_reward_shaping is not None:
                    #     r = self.player_reward_shaping(r, self.last_obs_act)
                    d = self.last_done
                    info = self.last_info
                    obs = np.concatenate([np.reshape(self.last_obs_act,masked_obs.shape),masked_obs])
                    obs = self.player_observation_space.convert(obs)
                else:
                    r = 0
                    info = (False)
                    obs = np.concatenate([np.reshape(self.ser_cum_act, self.last_masked_obs.shape),self.last_masked_obs])
                    if self.inc_player_last_act:
                        obs = np.append(obs, self.last_action)

                    obs = self.observer_observation_space.convert(obs)
                    d = False

            else:
                if not self.cont_act:
                    r_action = self.obs_action_translator(r_action, self.power_vec, self.obs_size)
                if self.fully_obs:
                    r_action = np.ones(r_action.shape)
                elif self.rand_obs:
                    r_action = np.random.randint(0,2,r_action.shape)
                self.player_turn = True
                self.last_obs_act = r_action
                masked_obs = np.multiply(np.reshape(r_action, self.last_obs.shape), self.last_obs)
                self.last_masked_obs = masked_obs
                info = self.last_info
                r = self.last_reward
                # if self.player_reward_shaping is not None:
                #     r = self.player_reward_shaping(r, r_action)
                d = self.last_done
                obs = np.concatenate([np.reshape(r_action, masked_obs.shape), masked_obs])
                obs = self.player_observation_space.convert(obs)

            return EnvStep(obs,r,d,info)


    def reset(self):
        self.curr_episode_length = 0
        self.last_done = False
        self.last_reward = 0
        self.last_action = self.player_action_space.revert(self.player_action_space.null_value())
        if self.serial:
            self.ser_cum_act = np.zeros(self.env.observation_space.shape)
            self.ser_counter = 0
        self.player_turn = False
        o = self.env.reset()
        self.last_obs = o
        obs = np.concatenate([np.zeros(o.shape),np.zeros(o.shape)])
        if self.inc_player_last_act:
            obs = np.append(obs,self.last_action) 
        self.last_obs_act = np.zeros(o.shape)
        self.last_masked_obs = np.zeros(o.shape)
        obs = self.observer_observation_space.convert(obs)
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

    def set_fully_observable(self):
        self.fully_obs = True

    def set_random_observation(self):
        self.rand_obs = True

