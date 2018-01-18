#!/usr/bin/env python3
import os, logging, gym
from baselines import logger
from baselines.common import set_global_seeds
from baselines import bench
from baselines.acktr.acktr_disc_flat import learn
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.acktr.policies import FcnPolicy
from baselines.common.atari_wrappers import (
    #NoopResetEnv,
    #MaxAndSkipEnv,
    #FireResetEnv,
    #WarpFrame,
    ClipRewardEnv,
    )

def train(env_id, num_timesteps, seed, num_cpu, save_interval=None, animate_interval=None):
    def make_env(rank):
        def _thunk():
            env = gym.make(env_id)
            env.seed(seed + rank)
            env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
            gym.logger.setLevel(logging.WARN)
            #env = NoopResetEnv(env, noop_max=30)
            #env = MaxAndSkipEnv(env, skip=4)
            #if 'FIRE' in env.unwrapped.get_action_meanings():
            #    env = FireResetEnv(env)
            #env = WarpFrame(env)
            env = ClipRewardEnv(env)
            return env
        return _thunk
    set_global_seeds(seed)
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    policy_fn = FcnPolicy
    learn(policy_fn, env, seed, total_timesteps=int(num_timesteps * 1.1), nprocs=num_cpu, save_interval=save_interval, animate_interval=animate_interval, env_id=env_id)
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='LunarLander-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    parser.add_argument('--save-interval', type=int, default=int(1000))
    parser.add_argument('--animate-interval', type=int, default=int(100))
    args = parser.parse_args()
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed, num_cpu=8, save_interval=args.save_interval, animate_interval=args.animate_interval)


if __name__ == '__main__':
    main()
