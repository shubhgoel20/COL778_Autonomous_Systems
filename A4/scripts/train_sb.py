#This file is a part of COL778 A4
import os
import cv2
import time
import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import gymnasium as gym

import torch
import panda_gym
import config as exp_config
import utils.utils as utils
import utils.pytorch_util as ptu
from utils.logger import Logger


def setup_agent(args, configs):
    
    from agents.mujoco_agents import SBAgent as Agent
    global agent, env
    agent = Agent(args.env_name, **configs['hyperparameters'])
    env = agent.env
    if args.load_checkpoint is not None:
        print('loading checkpoint from {}'.format(args.load_checkpoint))
        agent.load_checkpoint(args.load_checkpoint)
        print("Checkpoint loaded successfully")
    
def train_agent():
    agent.learn() #A timeout will be imposed on this function during evaluation. Make sure you save the model periodically to avoid losing progress.



    
    
def test_agent(args):
    vector_env = False
    env = gym.make(args.env_name, render_mode = 'rgb_array')
    obs = env.reset()
    is_gymnasium = True if isinstance(obs, tuple) else False
    
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data_test")
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    obs = obs[0] if (args.env_name == "PandaPush-v3" or is_gymnasium) else obs
    
    if  args.env_name != "PandaPush-v3" and obs.ndim >1:
        vector_env = True
    
    rewards = []
    image_obs =  []
    if args.env_name == "PandaPush-v3":
        sucess = []
    print("Started sampling trajectories using the agent")
    this_epi_rew = []
    num_steps = 2000 if args.env_name == "PandaPush-v3" else 10000
    for _ in range(num_steps):
        try:
            action, _ = agent.get_action(obs)
        except ValueError:
            action = agent.get_action(obs)
        except:
            raise ValueError("Error in getting action")
        
        if args.env_name == "PandaPush-v3" or is_gymnasium:
            obs, rew, terminated, truncated, info = env.step(action)
            if (truncated or terminated) and args.env_name == "PandaPush-v3":
                sucess.append(int(info["is_success"]))
            terminated = terminated or truncated
        else:
            obs, rew, terminated, info = env.step(action)
        
        _done = np.all(terminated) if vector_env else terminated
        
        if _done:
            obs = env.reset()
            obs = obs[0] if (args.env_name == "PandaPush-v3" or is_gymnasium) else obs
            rewards.append(np.sum(this_epi_rew))
            this_epi_rew = []
            continue
        
        if vector_env:
            render_img = env.get_images()
            render_img = render_img[0]
            rew = np.mean(rew)
        else:
            render_img = env.render()
                
        image_obs.append(render_img)
        this_epi_rew.append(rew)
    
    print(f"Mean reward: {np.mean(rewards)}")
    print(f"Success rate: {np.mean(sucess)}" if args.env_name == "PandaPush-v3" else "")
    with open(os.path.join(data_path, "rewards.txt"), "w") as f:
        print(f"{np.mean(rewards)}", file=f)
        print(f"{np.mean(sucess)}" if args.env_name == "PandaPush-v3" else " ", file=f)
    # Save the images as video

    height, width, layers = image_obs[0].shape
    size = (width,height)
    
    video_file_name = os.path.join(data_path, f"{args.env_name}_video.mp4")
    fps = 15 if args.env_name == 'PandaPush-v3' else 30
    out = cv2.VideoWriter(video_file_name,cv2.VideoWriter_fourcc(*'mp4v'),fps, (size[0], size[1]))
    print(f"Saving video at {data_path}")
    for i in range(len(image_obs)):
        out.write(image_obs[i])
    out.release()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--load_checkpoint', type=str)
    
    args = parser.parse_args()
    
    configs = exp_config.configs[args.env_name]['SB3']

    if args.test:
        import os
        for dir in os.listdir("./best_models"):
            dir_name = dir.lower()
            _env_name = args.env_name.lower()
            if _env_name in dir_name:
                args.load_checkpoint = os.path.join("./best_models", dir)
                break
        
        if args.load_checkpoint is None and len(os.listdir("./best_models"))>0:
            args.load_checkpoint = os.path.join("./best_models", os.listdir("./best_models")[-1])
            
        assert args.load_checkpoint is not None, "No checkpoint found for the environment. Please check if your model is getting saved properly"
        setup_agent(args, configs)
        test_agent(args)
    else:
        setup_agent(args, configs)
        train_agent()
            
        
    
    















