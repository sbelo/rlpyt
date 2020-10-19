from rlpyt.cwto_samplers.serial.sampler import SerialSampler
from rlpyt.cwto_samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.algos.pg.ppo import PPO
from rlpyt.cwto_runners.minibatch_rl import MinibatchRl, MinibatchRlEval
from rlpyt.utils.logging.context import logger_context
from rlpyt.spaces.int_box import IntBox
import os
import torch
import pickle
import gym
import whynot as wn
from gym.spaces.box import Box
from rlpyt.cwto_agents.cwto_agent_wrp import *
from rlpyt.cwto_envs.cwto_env_wrp import *
from rlpyt.cwto_agents.cwto_agent import CWTO_LstmAgent, CWTO_AlternatingLstmAgent
from rlpyt.cwto_models.cwto_model import CWTO_LstmModel
from gym.spaces import Discrete
from reward_shaping import observer_reward_shaping_cartpole,player_reward_shaping_cartpole,observer_reward_shaping_hiv,player_reward_shaping_hiv
from rlpyt.cwto_samplers.parallel.cpu.collectors import CpuEvalCollector
from rlpyt.cwto_samplers.serial.collectors import SerialEvalCollector
from rlpyt.utils.seed import set_envs_seeds,make_seed
from rlpyt.cwto_samplers.collections import TrajInfo_obs

def build_and_train(game="cartpole", run_ID=0, cuda_idx=None, sample_mode="serial", n_parallel=2, eval=False, serial=False, train_mask=[True,True],wandb_log=False,save_models_to_wandb=False, log_interval_steps=1e5, observation_mode="agent", inc_player_last_act=False, alt_train=False, eval_perf=False, n_steps=50e6):
    # def envs:
    if observation_mode == "agent":
        fully_obs = False
        rand_obs = False
    elif observation_mode == "random":
        fully_obs = False
        rand_obs = True
    elif observation_mode == "full":
        fully_obs = True
        rand_obs = False

    n_serial = None
    if game == "cartpole":
        work_env = gym.make
        env_name = 'CartPole-v1'
        state_space_low = np.asarray(
            [0.0, 0.0, 0.0, 0.0, -4.8000002e+00, -3.4028235e+38, -4.1887903e-01, -3.4028235e+38])
        state_space_high = np.asarray([1.0, 1.0, 1.0, 1.0, 4.8000002e+00, 3.4028235e+38, 4.1887903e-01, 3.4028235e+38])
        obs_space = Box(state_space_low, state_space_high, dtype=np.float32)
        player_act_space = work_env(env_name).action_space
        player_act_space.shape = (1,)
        print(player_act_space)
        if inc_player_last_act:
            observer_obs_space = Box(np.append(state_space_low,0), np.append(state_space_high,1), dtype=np.float32)
        else:
            observer_obs_space = obs_space
        player_reward_shaping = player_reward_shaping_cartpole
        observer_reward_shaping = observer_reward_shaping_cartpole
        max_decor_steps = 20
        b_size = 20
        num_envs = 16
        max_episode_length = np.inf
        player_model_kwargs = dict(hidden_sizes = [256,256],lstm_size = 256,nonlinearity = torch.nn.ReLU, normalize_observation = False, norm_obs_clip = 10, norm_obs_var_clip = 1e-6)
        observer_model_kwargs = dict(hidden_sizes = [256,256],lstm_size = 256,nonlinearity = torch.nn.ReLU, normalize_observation = False, norm_obs_clip = 10, norm_obs_var_clip = 1e-6)

    elif game == "hiv":
        work_env = wn.gym.make
        env_name = 'HIV-v0'
        state_space_low = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        state_space_high = np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
        obs_space = Box(state_space_low, state_space_high, dtype=np.float32)
        player_act_space = work_env(env_name).action_space
        if inc_player_last_act:
            observer_obs_space = Box(np.append(state_space_low,0), np.append(state_space_high,3), dtype=np.float32)
        else:
            observer_obs_space = obs_space
        player_reward_shaping = player_reward_shaping_hiv
        observer_reward_shaping = observer_reward_shaping_hiv
        max_decor_steps = 100
        b_size = 200
        num_envs = 16
        max_episode_length = 100
        player_model_kwargs = dict(hidden_sizes = [256,256],lstm_size = 256,nonlinearity = torch.nn.ReLU, normalize_observation = False, norm_obs_clip = 10, norm_obs_var_clip = 1e-6)
        observer_model_kwargs = dict(hidden_sizes = [256,256],lstm_size = 256,nonlinearity = torch.nn.ReLU, normalize_observation = False, norm_obs_clip = 10, norm_obs_var_clip = 1e-6)

    if serial:
        n_serial = int(len(state_space_high) / 2)
        observer_act_space = Discrete(2)
        observer_act_space.shape = (1,)
    else:
        observer_act_space = IntBox(low=0,high=int(2**int(len(state_space_high) / 2)))


    affinity = dict(cuda_idx=cuda_idx, workers_cpus=list(range(n_parallel)))
    gpu_cpu = "CPU" if cuda_idx is None else f"GPU {cuda_idx}"
    if sample_mode == "serial":
        alt = False
        Sampler = SerialSampler  # (Ignores workers_cpus.)
        if eval:
            eval_collector_cl = SerialEvalCollector
        else:
            eval_collector_cl = None
        print(f"Using serial sampler, {gpu_cpu} for sampling and optimizing.")
    elif sample_mode == "cpu":
        alt = False
        Sampler = CpuSampler
        if eval:
            eval_collector_cl = CpuEvalCollector
        else:
            eval_collector_cl = None
        print(f"Using CPU parallel sampler (agent in workers), {gpu_cpu} for optimizing.")
    env_kwargs = dict(work_env=work_env,env_name=env_name,obs_spaces=[obs_space,observer_obs_space],action_spaces=[player_act_space,observer_act_space],serial=serial,player_reward_shaping=player_reward_shaping,observer_reward_shaping=observer_reward_shaping,fully_obs=fully_obs,rand_obs=rand_obs,inc_player_last_act=inc_player_last_act,max_episode_length=max_episode_length)
    if eval:
        eval_env_kwargs = env_kwargs
        eval_max_steps = 1e4
        num_eval_envs = num_envs
    else:
        eval_env_kwargs = None
        eval_max_steps = None
        num_eval_envs = 0
    sampler = Sampler(
        EnvCls=CWTO_EnvWrapper,
        env_kwargs=env_kwargs,
        batch_T=b_size,
        batch_B=num_envs,
        max_decorrelation_steps=max_decor_steps,
        eval_n_envs = num_eval_envs,
        eval_CollectorCls = eval_collector_cl,
        eval_env_kwargs = eval_env_kwargs,
        eval_max_steps = eval_max_steps,
    )

    player_model = CWTO_LstmModel
    observer_model = CWTO_LstmModel

    player_algo = PPO()
    observer_algo = PPO()
    player = CWTO_LstmAgent(ModelCls=player_model, model_kwargs=player_model_kwargs, initial_model_state_dict=None)
    observer = CWTO_LstmAgent(ModelCls=observer_model, model_kwargs=observer_model_kwargs, initial_model_state_dict=None)
    agent = CWTO_AgentWrapper(player,observer,serial=serial,n_serial=n_serial,alt=alt, train_mask=train_mask)

    if eval:
        RunnerCl = MinibatchRlEval
    else:
        RunnerCl = MinibatchRl

    runner = RunnerCl(
        player_algo=player_algo,
        observer_algo=observer_algo,
        agent=agent,
        sampler=sampler,
        n_steps=n_steps,
        log_interval_steps=log_interval_steps,
        affinity=affinity,
        wandb_log=wandb_log,
        alt_train=alt_train
    )
    config = dict(domain=game)
    name = "ppo_" + game
    log_dir = os.getcwd() + "/cwto_logs/" + name
    with logger_context(log_dir, run_ID, name, config):
        runner.train()
    if save_models_to_wandb:
        agent.save_models_to_wandb()
    if eval_perf:
        eval_n_envs = 10
        eval_envs = [CWTO_EnvWrapper(**env_kwargs)
                     for _ in range(eval_n_envs)]
        set_envs_seeds(eval_envs, make_seed())
        eval_collector = SerialEvalCollector(
            envs=eval_envs,
            agent=agent,
            TrajInfoCls=TrajInfo_obs,
            max_T=1000,
            max_trajectories=10,
            log_full_obs=True
        )
        traj_infos_player, traj_infos_observer = eval_collector.collect_evaluation(runner.get_n_itr())
        observations = []
        player_actions = []
        returns = []
        observer_actions = []
        for traj in traj_infos_player:
            observations.append(np.array(traj.Observations))
            player_actions.append(np.array(traj.Actions))
            returns.append(traj.Return)
        for traj in traj_infos_observer:
            observer_actions.append(np.array([obs_action_translator(act, eval_envs[0].power_vec, eval_envs[0].obs_size) for act in traj.Actions]))

        # save results:
            open_obs = open('eval_observations.pkl',"wb")
            pickle.dump(observations,open_obs)
            open_obs.close()
            open_ret = open('eval_returns.pkl',"wb")
            pickle.dump(returns,open_ret)
            open_ret.close()
            open_pact = open('eval_player_actions.pkl',"wb")
            pickle.dump(player_actions,open_pact)
            open_pact.close()
            open_oact = open('eval_observer_actions.pkl',"wb")
            pickle.dump(observer_actions,open_oact)
            open_oact.close()

if __name__ == "__main__":
    import argparse
    import wandb

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--game', help='name of env', default='hiv', type=str, choices=['cartpole','hiv'])
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=None)
    parser.add_argument('--secondary_cuda_idx', help='gpu to use - for parallel work', type=int, default=None)
    parser.add_argument('--sample_mode', help='serial or parallel sampling',
                        type=str, default='cpu', choices=['serial', 'cpu'])
    parser.add_argument('--n_parallel', help='number of sampler workers', type=int, default=2)
    parser.add_argument('--eval', help='use evaluation runner', type=bool, default=False)
    parser.add_argument('--serial', help='serial observer', type=bool, default=False)
    parser.add_argument('--train_mask_ply', help='player train', type=bool, default=True)
    parser.add_argument('--train_mask_obs', help='observer train', type=bool, default=True)
    parser.add_argument('--wandb', help='wandb logging', type=bool, default=True)
    parser.add_argument('--wandb_project', help='wandb project name', type=str, default="choose_what_to_observe_rlpyt")
    parser.add_argument('--wandb_run_name', help='wandb run name', type=str, default=None)
    parser.add_argument('--wandb_save_models', help='save models to wandb', type=bool, default=False)
    parser.add_argument('--log_interval_steps', help='interval between logs', type=int, default=1e5)
    parser.add_argument('--obs_mode', help='choice of observations', default='agent', type=str, choices=['agent','random','full'])
    parser.add_argument('--inc_player_last_act', help='include player action in observer observation', type=bool, default=False)
    parser.add_argument('--alt_train', help='each time only one agent optimized', type=bool, default=False)
    parser.add_argument('--eval_perf', help='evaluate the performance of the player', type=bool, default=False)
    parser.add_argument('--n_steps', help='number of optimization steps to run', type=int, default=50e6)
    args = parser.parse_args()
    if args.wandb:
        wandb.init(project=args.wandb_project,name=args.wandb_run_name)

    if args.secondary_cuda_idx is not None:
        assert args.cuda_idx is not None
        cuda_idxs = [args.cuda_idx, args.secondary_cuda_idx]
    else:
        cuda_idxs = args.cuda_idx

    build_and_train(
        game=args.game,
        run_ID=args.run_ID,
        cuda_idx=cuda_idxs,
        sample_mode=args.sample_mode,
        n_parallel=args.n_parallel,
        eval=args.eval,
        serial=args.serial,
        train_mask=[args.train_mask_ply,args.train_mask_obs],
        wandb_log = args.wandb,
        save_models_to_wandb=args.wandb_save_models,
        log_interval_steps=args.log_interval_steps,
        observation_mode=args.obs_mode,
        inc_player_last_act=args.inc_player_last_act,
        alt_train=args.alt_train,
        eval_perf=args.eval_perf,
        n_steps=args.n_steps
    )
