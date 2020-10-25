# ref: example_2.py
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.envs.gym import make as gym_make
from rlpyt.algos.qpg.sac import SAC
from rlpyt.agents.qpg.sac_agent import SacAgent
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging.context import logger_context
from rlpyt.samplers.parallel.cpu.collectors import CpuEvalCollector
from rlpyt.samplers.serial.collectors import SerialEvalCollector

def build_and_train(env_id="HalfCheetah-v2", run_ID=0, cuda_idx=None,sample_mode='cpu',n_parallel=2,eval=False,wandb_log=True,log_interval_steps=1e5,n_steps=50e6):
    affinity = dict(cuda_idx=cuda_idx, workers_cpus=list(range(n_parallel)))
    gpu_cpu = "CPU" if cuda_idx is None else f"GPU {cuda_idx}"
    if sample_mode == "serial":
        Sampler = SerialSampler  # (Ignores workers_cpus.)
        if eval:
            eval_collector_cl = SerialEvalCollector
        else:
            eval_collector_cl = None

        print(f"Using serial sampler, {gpu_cpu} for sampling and optimizing.")
    elif sample_mode == "cpu":
        Sampler = CpuSampler
        if eval:
            eval_collector_cl = CpuEvalCollector
        else:
            eval_collector_cl = None

        print(f"Using CPU parallel sampler (agent in workers), {gpu_cpu} for optimizing.")

     
    num_envs = 8
    b_size = 20
     
    env_kwargs = dict(id=env_id)
    if eval:
        eval_env_kwargs = env_kwargs
        eval_max_steps = 1e4
        num_eval_envs = num_envs
    else:
        eval_env_kwargs = None
        eval_max_steps = None
        num_eval_envs = 0

    sampler = Sampler(
        EnvCls=gym_make,
        env_kwargs=env_kwargs,
        eval_env_kwargs=eval_env_kwargs,
        batch_T=b_size,  # One time-step per sampler iteration.
        batch_B=num_envs,  # One environment (i.e. sampler Batch dimension).
        max_decorrelation_steps=0,
        eval_n_envs=num_eval_envs,
        eval_CollectorCls = eval_collector_cl,
        eval_max_steps=eval_max_steps,
    )
#         eval_max_trajectories=50,
#     )
    algo = SAC()  # Run with defaults.
    agent = SacAgent()
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=n_steps,
        log_interval_steps=log_interval_steps,
        affinity=dict(cuda_idx=cuda_idx),
        wandb_log=True
    )
    config = dict(env_id=env_id)
    name = "sac_" + env_id
    log_dir = "baseline_run"
    with logger_context(log_dir, run_ID, name, config):
        runner.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--env_id', help='name of env', default='HalfCheetah-v2', type=str)
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=None)
    parser.add_argument('--sample_mode', help='serial or parallel sampling', type=str, default='cpu', choices=['serial', 'cpu'])
    parser.add_argument('--n_parallel', help='number of sampler workers', type=int, default=2)
    parser.add_argument('--eval', help='use evaluation runner', type=bool, default=False)
    parser.add_argument('--wandb', help='wandb logging', type=bool, default=False)
    parser.add_argument('--wandb_project', help='wandb project name', type=str, default="choose_what_to_observe_rlpyt")
    parser.add_argument('--wandb_run_name', help='wandb run name', type=str, default=None)
    parser.add_argument('--wandb_group', help='wandb group name', type=str, default=None)
    parser.add_argument('--log_interval_steps', help='interval between logs', type=int, default=1e5)
    parser.add_argument('--n_steps', help='number of optimization steps to run', type=int, default=50e6)
    args = parser.parse_args()
    
    if args.wandb:
        wandb.init(project=args.wandb_project,group=args.wandb_group,name=args.wandb_run_name)
    
    build_and_train(
        env_id=args.env_id,
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
        sample_mode=args.sample_mode,
        n_parallel=args.n_parallel,
        eval=args.eval,
        wandb_log=args.wandb,
        log_interval_steps=args.log_interval_steps,
        n_steps=args.n_steps
    )
