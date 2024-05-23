import gymnasium as gym
import car_gym
from car_gym.envs.hybrid_car.car import generate_random_map
import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np


SMALL_ENV = gym.make('HybridCar-v1', desc = generate_random_map(20, 0.95, 0),  render_mode = 'rgb_array')
LARGE_ENV = gym.make('HybridCar-v1', desc = generate_random_map(40, 0.95, 0),  render_mode = 'rgb_array')


is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

def plot_rewards(rewards, show_result=False):
    '''
    rewards: list of rewards to be plotted. 
    '''
    plt.figure(1)
    
    durations_t = torch.tensor(rewards, dtype=torch.float)
    if show_result:
        plt.title('Trained')
    else:
        plt.clf()
        plt.title('Training...')
    
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.plot(durations_t.numpy())
    
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
