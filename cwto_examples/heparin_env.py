from gym import Env
from gym.spaces import Discrete,Box
import random
import numpy as np
from gym.wrappers.time_limit import TimeLimit

model_dir = 'model2'
class HeparinEnv(Env):
    action_space = None
    observation_space = None

    def __init__(self,env_name):
        self.Trans = np.load(model_dir + "/Transitions.npy")
        self.Feat = np.load(model_dir + "/Features.npy")
        self.Cont = np.load(model_dir + "/Contexts.npy")
        self.Rew = np.load(model_dir + "/Rewards.npy")
        self.InitCont = np.load(model_dir + "/InitCont.npy")
        self.InitStat = np.load(model_dir + "/InitStat.npy")
        self.nstates = self.Rew.size
        self.action_space = Discrete(self.Trans.shape[2])
        self.observation_space = Box(np.asarray([np.min(self.Cont[:,i]) for i in range(len(self.Cont[0]))] + [np.min(self.Feat[:,i]) for i in range(len(self.Feat[0]))]),np.asarray([np.max(self.Cont[:,i]) for i in range(len(self.Cont[0]))] + [np.max(self.Feat[:,i]) for i in range(len(self.Feat[0]))]),dtype=np.float32)
        # g = 0
        # self.legal_cs = []
        # while True:
        #     try:
        #         cs = np.load("model/LegalCS_" + str(g) + ".npy")
        #     except:
        #         break
        #     self.legal_cs.append(cs)
        #
        #     g += 1
        random.seed()
        np.random.seed()
        self.curr_st = 0
        self.env = TimeLimit(self,50)


    def step(self, action):
        self.curr_state = np.random.choice(np.arange(self.nstates),p=self.Trans[self.curr_cont,self.curr_state,action,:])
        self.curr_st += 1
        reward = self.Rew[self.curr_state]
        feat = self.Feat[self.curr_state]
        feat = np.reshape(feat,feat.size)
        if self.curr_st > 10 and self.curr_state == self.nstates - 1:
            done = True
            reward = 0
        else:
            done = False

        obs = np.concatenate([self.ccont,feat],axis=0)
        # print(obs)
        return np.reshape(obs,obs.size), reward, done, {'timeout':done}

    def reset(self):
        self.curr_st = 0
        self.curr_cont = np.random.choice(np.arange(len(self.InitCont)),p=self.InitCont) #random.randint(0,len(self.legal_cs)-1)
        self.curr_state = np.random.choice(np.arange(self.nstates),p=self.InitStat[self.curr_cont]) #self.legal_cs[self.curr_cont][random.randint(0,len(self.legal_cs[self.curr_cont])-1)]
        self.ccont = self.Cont[self.curr_cont]
        self.ccont = np.reshape(self.ccont,self.ccont.size)
        feat = self.Feat[self.curr_state]
        feat = np.reshape(feat,feat.size)
        obs = np.concatenate([self.ccont,feat],axis=0)
        # print(obs)
        return np.reshape(obs,obs.size)

    def render(self, mode='human'):
        pass

    def close(self):
        self.Trans = None
        self.Feat = None
        self.Cont = None
        self.Rew = None
        self.nstates = None
        self.legal_cs = None

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)