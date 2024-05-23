#This file is a part of COL864 A2
import os
import time

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))


import gym
import numpy as np
import torch

import cv2
import config as exp_config
import utils.utils as utils
import utils.pytorch_util as ptu
from utils.logger import Logger

MAX_NVIDEO = 2

def setup_agent(args, configs, load_checkpoint):
    print("loading checkpoint", load_checkpoint)
    global env, agent
    
    env = gym.make(args.env_name,render_mode=None)
    env.action_space.seed()
    env.reset()
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    if args.exp_name == "imitation":
        from agents.mujoco_agents import ImitationAgent as Agent    
    elif args.exp_name == "RL":
        from agents.mujoco_agents import RLAgent as Agent
    elif args.exp_name == "imitation-RL":
        from agents.mujoco_agents import ImitationSeededRL as Agent
    else:
        raise ValueError(f"Invalid experiment name {args.exp_name}")

    agent = Agent(ob_dim, ac_dim, args, **configs['hyperparameters'])
    if load_checkpoint is not None:
        state_dict = torch.load(load_checkpoint)
        state_dict = {key: value for key, value in state_dict.items() if "expert_policy" not in key}
        agent.load_state_dict(state_dict)


def test_agent(args):
    if hasattr(env, "model"):
        fps = 1 / env.model.opt.timestep
    else:
        fps = env.env.metadata["render_fps"]

    logger = Logger(args.logdir)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    total_envsteps = 0
    start_time = time.time()

    agent.to(ptu.device)

    eval_trajs, _ = utils.sample_trajectories(env, agent.get_action, 50000, 1000)
    logs = utils.compute_eval_metrics(eval_trajs)

    with open(os.path.join(args.logdir, "rewards.txt"), "w") as f:
        print(logs['Final_Eval_AverageReturn'], file=f)
    
    
    
    for key, value in logs.items():
        print("{} : {}".format(key, value))
        logger.log_scalar(value, key, 0)
    print("Done logging eval...\n\n")

    print("\nCollecting video rollouts...")
    eval_video_trajs = utils.sample_n_trajectories(
        env, agent.get_action, MAX_NVIDEO, 1000, render=True
    )

    is_logged_video = logger.log_trajs_as_videos(
        eval_video_trajs,
        1,
        fps=fps,
        max_videos_to_save=MAX_NVIDEO,
        video_title="eval_rollouts",
    )
    
    #save the videos to args.logdir
    videos = [traj['image_obs'] for traj in eval_video_trajs]
    for i, video in enumerate(videos):
        height, width, layers = video[0].shape
        size = (width,height)
        out = cv2.VideoWriter(f'{args.logdir}/{args.env_name}_video_{i}.mp4',cv2.VideoWriter_fourcc(*'mp4v'),20,(size[0], size[1]))
        for frame in video:
            out.write(frame)
        out.release()
        


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--exp_name", type=str, choices = ["imitation", "RL"], required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-gpu_id", default=0)
    parser.add_argument("--video_log_freq", type=int, default=-1)
    parser.add_argument("--scalar_log_freq", type=int, default=1)
    parser.add_argument("--load_checkpoint", type=str, default=None)
    # parser.add_argument("--logdir_name", type=str, default="data")

    args = parser.parse_args()

    configs = exp_config.configs[args.env_name][args.exp_name]

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data_test")
    model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../models")
    
    if os.path.exists(data_path):
        os.system(f"rm -r {data_path}")
    
    os.makedirs(data_path)

    # logdir = (
    #     args.exp_name
    #     + "_"
    #     + args.env_name
    #     + "_"
    #     + time.strftime("%d-%m-%Y_%H-%M-%S")
    # )
    logdir = data_path
    args.logdir = logdir
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print("\n\n =====================================================")
    print("Testing agent..........")
    
    checkpoint_path = None
    for dir in os.listdir("./best_models"):
        env_name = args.env_name.lower()
        exp_name = args.exp_name.lower()
        dir_name = dir.lower()
        if env_name in dir_name and exp_name in dir_name:
            checkpoint_path = os.path.join("./best_models", dir)
    
    if checkpoint_path is None and len(os.listdir("./best_models"))>0:
        checkpoint_path = os.path.join("./best_models", os.listdir("./best_models")[-1])
    
    if checkpoint_path is not None:
        setup_agent(args, configs, checkpoint_path)

        test_agent(args)
    else:
        print("No checkpoint found for the given env and exp name")
        print("Please train the model first")
        raise ValueError("No best model saved found. Please check if your model is getting saved properly")
    

if __name__ == "__main__":
    main()
