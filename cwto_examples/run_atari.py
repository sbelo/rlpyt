from rlpyt.cwto_samplers.serial.sampler import SerialSampler
from rlpyt.cwto_samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.cwto_runners.minibatch_rl import MinibatchRl, MinibatchRlEval
from rlpyt.utils.logging.context import logger_context
import os
from rlpyt.cwto_agents.cwto_agent_atari import CWTO_AtariFfAgent, CWTO_AtariLstmAgent
from rlpyt.cwto_agents.cwto_agent_wrp import *
from rlpyt.cwto_envs.cwto_env_wrp_atari import *
from rlpyt.cwto_samplers.parallel.cpu.collectors import CpuEvalCollector
from rlpyt.cwto_samplers.serial.collectors import SerialEvalCollector
from rlpyt.algos.pg.a2c import A2C
from rlpyt.algos.qpg.sac import SAC
from rlpyt.agents.pg.atari import AtariLstmAgent

def build_and_train(windx,windy,game="pong", run_ID=0, cuda_idx=None, sample_mode="serial", n_parallel=2, num_envs=2, eval=False, train_mask=[True,True],wandb_log=False,save_models_to_wandb=False, log_interval_steps=1e5, alt_train=False, n_steps=50e6,max_episode_length=np.inf,b_size=5,max_decor_steps=10):
    # def envs:
    # player_model_kwargs = dict(hidden_sizes=[32], lstm_size=16, nonlinearity=torch.nn.ReLU, normalize_observation=False,
    #                            norm_obs_clip=10, norm_obs_var_clip=1e-6)
    # observer_model_kwargs = dict(hidden_sizes=[128], lstm_size=16, nonlinearity=torch.nn.ReLU,
    #                              normalize_observation=False, norm_obs_clip=10, norm_obs_var_clip=1e-6)
    player_reward_shaping = None
    observer_reward_shaping = None
    window_size = np.asarray([windx,windy])

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
    env_kwargs = dict(env_name=game,window_size=window_size,player_reward_shaping=player_reward_shaping,observer_reward_shaping=observer_reward_shaping,max_episode_length=max_episode_length)
    if eval:
        eval_env_kwargs = env_kwargs
        eval_max_steps = 1e4
        num_eval_envs = num_envs
    else:
        eval_env_kwargs = None
        eval_max_steps = None
        num_eval_envs = 0
    sampler = Sampler(
        EnvCls=CWTO_EnvWrapperAtari,
        env_kwargs=env_kwargs,
        batch_T=b_size,
        batch_B=num_envs,
        max_decorrelation_steps=max_decor_steps,
        eval_n_envs = num_eval_envs,
        eval_CollectorCls = eval_collector_cl,
        eval_env_kwargs = eval_env_kwargs,
        eval_max_steps = eval_max_steps,
    )


    player_algo = A2C()
    observer_algo = A2C()
    player = AtariLstmAgent() #model_kwargs=player_model_kwargs)
    observer = CWTO_AtariLstmAgent() #model_kwargs=observer_model_kwargs)
    agent = CWTO_AgentWrapper(player,observer,alt=alt, train_mask=train_mask)

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
    log_dir = os.getcwd() + "/cwto_logs/" + game
    with logger_context(log_dir, run_ID, game, config):
        runner.train()
    if save_models_to_wandb:
        agent.save_models_to_wandb()

if __name__ == "__main__":
    import argparse
    import wandb

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--game', help='name of env', default='pong', type=str)
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=None)
    parser.add_argument('--secondary_cuda_idx', help='gpu to use - for parallel work', type=int, default=None)
    parser.add_argument('--sample_mode', help='serial or parallel sampling',
                        type=str, default='cpu', choices=['serial', 'cpu'])
    parser.add_argument('--n_parallel', help='number of sampler workers', type=int, default=2)
    parser.add_argument('--num_envs', help='number of parallel envs', type=int, default=2)
    parser.add_argument('--eval', help='use evaluation runner', type=bool, default=False)
    parser.add_argument('--train_mask_ply', help='player train', type=bool, default=True)
    parser.add_argument('--train_mask_obs', help='observer train', type=bool, default=True)
    parser.add_argument('--wandb', help='wandb logging', type=bool, default=False)
    parser.add_argument('--wandb_project', help='wandb project name', type=str, default="choose_what_to_observe_atari")
    parser.add_argument('--wandb_run_name', help='wandb run name', type=str, default=None)
    parser.add_argument('--wandb_group', help='wandb group name', type=str, default=None)
    parser.add_argument('--wandb_save_models', help='save models to wandb', type=bool, default=False)
    parser.add_argument('--log_interval_steps', help='interval between logs', type=int, default=1e5)
    parser.add_argument('--alt_train', help='each time only one agent optimized', type=bool, default=False)
    parser.add_argument('--n_steps', help='number of optimization steps to run', type=int, default=50e6)
    parser.add_argument('--max_episode_len', help='maximal episode length', type=float, default=np.inf)
    parser.add_argument('--b_size', help='batch size', type=int, default=5)
    parser.add_argument('--max_decor', help='maximal number of decorrelation steps', type=int, default=10)
    parser.add_argument('--windx', help='x axis size of window', type=int)
    parser.add_argument('--windy', help='y axis size of window', type=int)
    args = parser.parse_args()
    if args.wandb:
        wandb.init(project=args.wandb_project,group=args.wandb_group,name=args.wandb_run_name)

    if args.secondary_cuda_idx is not None:
        assert args.cuda_idx is not None
        cuda_idxs = [args.cuda_idx, args.secondary_cuda_idx]
    else:
        cuda_idxs = args.cuda_idx

    build_and_train(
        windx=args.windx,
        windy=args.windy,
        game=args.game,
        run_ID=args.run_ID,
        cuda_idx=cuda_idxs,
        sample_mode=args.sample_mode,
        n_parallel=args.n_parallel,
        eval=args.eval,
        train_mask=[args.train_mask_ply,args.train_mask_obs],
        wandb_log = args.wandb,
        save_models_to_wandb=args.wandb_save_models,
        log_interval_steps=args.log_interval_steps,
        alt_train=args.alt_train,
        n_steps=args.n_steps,
        num_envs=args.num_envs,
        max_episode_length=args.max_episode_len,
        b_size=args.b_size,
        max_decor_steps=args.max_decor
    )
