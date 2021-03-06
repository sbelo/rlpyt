
from collections import namedtuple

from rlpyt.utils.collections import namedarraytuple, AttrDict
import numpy as np

Samples = namedarraytuple("Samples", ["agent", "env"])

AgentSamples = namedarraytuple("AgentSamples",
    ["action", "prev_action", "agent_info"])
AgentSamplesBsv = namedarraytuple("AgentSamplesBsv",
    ["action", "prev_action", "agent_info", "bootstrap_value"])
EnvSamples = namedarraytuple("EnvSamples",
    ["observation", "reward", "prev_reward", "done", "env_info"])


class BatchSpec(namedtuple("BatchSpec", "T B")):
    """
    T: int  Number of time steps, >=1.
    B: int  Number of separate trajectory segments (i.e. # env instances), >=1.
    """
    __slots__ = ()

    @property
    def size(self):
        return self.T * self.B


class TrajInfo(AttrDict):
    """
    Because it inits as an AttrDict, this has the methods of a dictionary,
    e.g. the attributes can be iterated through by traj_info.items()
    Intent: all attributes not starting with underscore "_" will be logged.
    (Can subclass for more fields.)
    Convention: traj_info fields CamelCase, opt_info fields lowerCamelCase.
    """

    _discount = 1  # Leading underscore, but also class attr not in self.__dict__.

    def __init__(self, n_obs=None, serial=False, **kwargs):
        super().__init__(**kwargs)  # (for AttrDict behavior)
        self.Length = 0
        self.Return = 0
        self.NonzeroRewards = 0
        self.DiscountedReturn = 0
        self._cur_discount = 1
        self.TotalCost = 0
        self._atari = False
        if n_obs is not None:
            if hasattr(n_obs,'__iter__'):
                self._atari = True
                self._window_size = n_obs
                self._null_flag = True
            else:
                self._serial = serial
                self._n_obs = n_obs
                for i in range(n_obs):
                    setattr(self,"ObsPercentFeature" + str(i+1),0)
                self.OverAllObsPercent = 0


    def step(self, observation, action, reward, done, agent_info, env_info, cost=0, obs_act=None):
        self.Length += 1
        self.Return += reward
        self.NonzeroRewards += reward != 0
        self.DiscountedReturn += self._cur_discount * reward
        self._cur_discount *= self._discount
        self.TotalCost += cost
        if obs_act is not None:
#             assert np.array_equal(obs_act[0],obs_act[1]) and np.array_equal(obs_act[2],obs_act[3]) and np.array_equal(obs_act[0],obs_act[2])
            if self._atari:
                if self._null_flag:
                    x_res = int(np.ceil(observation.shape[1] / self._window_size[0]))
                    y_res = int(np.ceil(observation.shape[2] / self._window_size[1]))
                    self._masks = np.zeros([x_res,y_res,observation.shape[1],observation.shape[2]],dtype=bool)
                    zeromask = np.zeros([observation.shape[1],observation.shape[2]],dtype=bool)
                    for i in range(x_res):
                        xmask = zeromask.copy()
                        if i == x_res - 1:
                            xmask[i*self._window_size[0]:-1,:] = True
                        else:
                            xmask[i*self._window_size[0]:(i+1)*self._window_size[0],:] = True
                        for j in range(y_res):
                            ymask = zeromask.copy()
                            if j == y_res - 1:
                                ymask[j*self._window_size[1]:-1,:] = True
                            else:
                                ymask[j*self._window_size[1]:(j+1)*self._window_size[1],:] = True
                            self._masks[i,j] = np.bitwise_and(xmask,ymask)
                            setattr(self,"ObsMap" + str(i) + "x" + str(j),0)
                for i in range(self._masks.shape[0]):
                    for j in range(self._masks.shape[1]):
                        setattr(self,"ObsMap" + str(i) + "x" + str(j),getattr(self,"ObsMap" + str(i) + "x" + str(j)) + np.sum(obs_act[0][self._masks[i,j]]))
            else:
                self.OverAllObsPercent += np.sum(obs_act) / self._n_obs
                for i in range(self._n_obs):
                    setattr(self,"ObsPercentFeature" + str(i+1),getattr(self,"ObsPercentFeature" + str(i+1)) + obs_act[i])

    def terminate(self, observation):
        if hasattr(self,"OverAllObsPercent"):
            if self._serial:
                length = self.Length / self._n_obs
            else:
                length = self.Length
            self.OverAllObsPercent = 100*self.OverAllObsPercent / length
            for i in range(self._n_obs):
                setattr(self, "ObsPercentFeature" + str(i + 1), 100*getattr(self, "ObsPercentFeature" + str(i + 1)) / length)
        elif self._atari:
            tot_sum = 0
            for i in range(self._masks.shape[0]):
                for j in range(self._masks.shape[1]):
                    tot_sum += getattr(self,"ObsMap" + str(i) + "x" + str(j))
            for i in range(self._masks.shape[0]):
                for j in range(self._masks.shape[1]):
                    setattr(self,"ObsMap" + str(i) + "x" + str(j),(1.0/tot_sum) * getattr(self,"ObsMap" + str(i) + "x" + str(j)))
        return self
        
class TrajInfo_obs(TrajInfo):
    """
    Because it inits as an AttrDict, this has the methods of a dictionary,
    e.g. the attributes can be iterated through by traj_info.items()
    Intent: all attributes not starting with underscore "_" will be logged.
    (Can subclass for more fields.)
    Convention: traj_info fields CamelCase, opt_info fields lowerCamelCase.
    """

    _discount = 1  # Leading underscore, but also class attr not in self.__dict__.

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # (for AttrDict behavior)
        self.Observations = []
        self.Actions = []
      

    def step(self, observation, action, reward, done, agent_info, env_info, cost=0, obs_act=None):
        super().step(observation, action, reward, done, agent_info, env_info)
        self.Observations.append(observation)
        self.Actions.append(action)

    def terminate(self, observation):
        return self

