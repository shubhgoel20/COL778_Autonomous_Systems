import gymnasium as gym
import car_gym
from utils import SMALL_ENV, LARGE_ENV
from utils import plot_rewards
import matplotlib.pyplot as plt
import time
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from tqdm import tqdm
import torch.nn.init as init


#Example USAGE
'''
env can be SMALL_ENV or LARGE_ENV

reset the environment: 
        observation, info = env.reset()
Next state on taking an action: 
        observation, reward, terminated, truncated, info = env.step(action)
Render the current_state:
    rgb_image = env.render

For more information about the gym API please visit https://gymnasium.farama.org/api/env/
New to OpenAI gym ?  Get started here https://gymnasium.farama.org/content/basic_usage/
'''

## Keep your Tabular Q-Learning implementatio here

class NN(nn.Module):
        def __init__(self, grid_size = 20, inp_channels = 40, hidden_size = 64, num_layers = 2, num_actions = 4):
                super(NN, self).__init__()
                linear_layers = []
                linear_layers.append(nn.Linear(inp_channels,hidden_size))
                linear_layers.append(nn.Tanh())
                for i in range(num_layers):
                        linear_layers.append(nn.Linear(hidden_size,hidden_size))
                        linear_layers.append(nn.Tanh())
                linear_layers.append(nn.Linear(hidden_size,num_actions))

                self.seq_model = nn.Sequential(*linear_layers)
                self.grid_size = grid_size

            # Initialize layers with Xavier initialization
                for layer in self.seq_model:
                        if isinstance(layer, nn.Linear):
                                init.xavier_uniform_(layer.weight)
                                if layer.bias is not None:
                                        init.constant_(layer.bias, 0)
        
        def forward(self, state):

                bs = state.shape[0]
                indices = np.arange(bs).astype(np.int32)

                row = state//self.grid_size
                col = state%self.grid_size

                row_vector = np.zeros((bs, self.grid_size))
                row_vector[indices, row.detach().numpy().astype(np.int32)] = 1.0

                assert row_vector.shape[0] == bs

                col_vector = np.zeros((bs, self.grid_size))
                col_vector[indices, col.detach().numpy().astype(np.int32)] = 1.0

                x = torch.cat([torch.FloatTensor(row_vector),torch.FloatTensor(col_vector)], dim =1)
                assert x.shape[0] == bs
                # print(x.dtype)
                x = self.seq_model(x)

                return x


class DQLearning:
        def __init__(self, env, alpha, gamma, eps, num_episodes, grid_size = 20, buffer_size = 10000, bs = 256, target_net_update = 25):
                self.env = env
                self.alpha = alpha #Learning Rate
                self.gamma = gamma #discounting factor
                self.eps = eps #epsilon-greedy
                self.num_episodes = num_episodes
                # self.learned_policy = np.zeros(env.observation_space.n)
                self.num_states = env.observation_space.n
                self.num_actions = env.action_space.n
                    
                self.inp_dim = 2*grid_size
                self.replay_buffer_cap = buffer_size #buffer capacity
                self.batch_size = bs #batch_size
                self.target_net_update = target_net_update #time delayed update condition
                self.counter = 0
                self.replay_buffer=deque(maxlen=buffer_size) #replay buffer

                self.target_model = NN(grid_size=grid_size, inp_channels=self.inp_dim)
                self.online_model = NN(grid_size=grid_size, inp_channels=self.inp_dim)
                # self.online_model = torch.load('dqn_large.pth')

                self.target_model.load_state_dict(self.online_model.state_dict())
                self.grid_size = grid_size
                self.start_state_q_value = []
                self.reward_list = []

                # self.actionsAppend=[]

        def best_action(self, state):
                q_values = self.online_model(torch.FloatTensor([state])).view(-1)
                assert q_values.shape[0] == 4
                q_values = q_values.detach().numpy()
                return np.argmax(q_values)
        
        def best_value(self, state):
                q_values = self.online_model(torch.FloatTensor([state])).view(-1)
                assert q_values.shape[0] == 4
                q_values = q_values.detach().numpy()
                return np.max(q_values)

        def select_action(self, state, episode_no):
                random_num = np.random.random()
                if random_num < self.eps:
                        return np.random.choice(self.num_actions)
                else:
                        return self.best_action(state)

        def simulate(self):
                for episode in tqdm(range(self.num_episodes)):
                        obs, info = self.env.reset()
                        terminated = False
                        truncated = False
                        sum_reward = 0
                        while not (terminated or truncated):
                                action = self.select_action(obs, episode)
                                next_obs, reward, terminated, truncated, info = self.env.step(action)
                                next_action = self.best_action(next_obs)
                                sum_reward = reward + self.gamma*sum_reward
                                self.replay_buffer.append((obs,action,reward, next_obs, next_action, terminated))
                                self.train_model()
                                obs = next_obs

                        self.start_state_q_value.append(self.best_value(0))
                        self.reward_list.append(sum_reward)
                        
                        if episode > 2500:
                                if self.eps > 0.5: self.eps = 0.9999*self.eps
                                
                        # if episode%50 == 0:
                        #         v = np.zeros(n*n)
                        #         policy = np.zeros(n*n)

                        #         for i in range(n*n):
                        #                 q_values = self.online_model(torch.FloatTensor([i])).view(-1)
                        #                 q_values = q_values.detach().numpy()
                        #                 action = np.argmax(q_values)
                        #                 policy[i] = action
                        #                 v[i] = np.max(q_values)


                        #         heat_map(v, policy, n)

        def train_model(self):
                # print("Buffer Size: ", len(self.replay_buffer))
                if (len(self.replay_buffer)> self.batch_size):
                        # print("Buffer Ready")
                        batch = random.sample(self.replay_buffer, self.batch_size)
                        states = []
                        actions = []
                        next_states = []
                        next_actions = []
                        rewards = []
                        mask = []
                        for obs,action,reward, next_obs, next_action, terminated in batch:
                                states.append(obs)
                                actions.append(action)
                                next_states.append(next_obs)
                                next_actions.append(next_action)
                                rewards.append(reward)
                                if terminated:
                                        mask.append(0)
                                else:
                                        mask.append(1)    
                        states = torch.FloatTensor(states)
                        next_states = torch.FloatTensor(next_states)
                        rewards = torch.FloatTensor(rewards)
                        mask = torch.FloatTensor(mask)
                        indices = np.arange(self.batch_size).astype(np.int32)

                        y_pred = self.online_model(states)[indices, actions]
                        y_target = rewards + self.gamma*self.target_model(next_states)[indices,next_actions]*mask

                        loss_fn = nn.MSELoss()
                        optimizer = optim.Adam(self.online_model.parameters(), lr=1e-5)
                        
                        optimizer.zero_grad()

                        loss = loss_fn(y_pred, y_target)
                        loss.backward()
                        optimizer.step()
                        self.counter+=1
                        if self.counter >= self.target_net_update:
                                # print("Updating target model")
                                self.target_model.load_state_dict(self.online_model.state_dict())
                                self.counter = 0

                        
        
        # def get_policy(self):
        #         for state in range(self.num_states):
        #                 self.learned_policy[state]= np.random.choice(np.where(self.Qmat[state]==np.max(self.Qmat[state]))[0])

        def plot_start_state(self, env_type):
                plt.plot(self.start_state_q_value, marker = 'o', markersize = 3)
                plt.xlabel('Episode Number')
                plt.ylabel('Start State Q Value')
                plt.title(env_type)
                plt.grid(True, ls = '--')
                plt.savefig(env_type+"_nn_values.png")
                plt.close()
        
        def plot_rewards(self, env_type):
                plt.plot(self.reward_list, marker = 'o', markersize = 3)
                plt.xlabel('Episode Number')
                plt.ylabel('Discounted Rewards Sum')
                plt.title(env_type)
                plt.grid(True, ls = '--')
                plt.savefig(env_type+"_nn_rewards.png")
                plt.close()


dict = {0:np.array([-1,0], dtype=float), 
        1 : np.array([0,1], dtype=float), 
        2 : np.array([1,0], dtype=float), 
        3 : np.array([0,-1], dtype=float)}

def heat_map(v,policy,n):
        plt.ioff()
        v = v.reshape(n, n)
        policy = policy.reshape(n, n)
        directions = np.zeros((n,n,2))
        for i in range(n):
                for j in range(n):
                        directions[i, j, :] = dict[int(policy[i, j])]

        Y, X = np.mgrid[0:n, 0:n]
        Y = Y - directions[:,:,1]/2
        X = X - directions[:,:,0]/2
        U = directions[:, :, 0]
        V = directions[:, :, 1]

        # Create figure and axes
        fig, ax = plt.subplots()

        # Plot heatmap
        cax = ax.imshow(v, cmap='hot', interpolation='nearest')

        # Plot arrows
        ax.quiver(X, Y, U, V, color='blue', angles='xy', scale_units='xy', scale=1.1)

        # Add colorbar
        cbar = fig.colorbar(cax)
        plt.savefig('heat_map_nn_final_large.png')
        plt.ion()
        plt.close()
        # Show plot
        # plt.show()

#TODO

seed_value = 42  # You can use any integer value
np.random.seed(seed_value)
env = LARGE_ENV
alpha = 0.05 #not needed
gamma = 0.99
eps = 1
num_episodes = 20000
n = 40
model = DQLearning(env,alpha,gamma,eps,num_episodes, grid_size=n)
model.simulate()
torch.save(model.online_model, "dqn_large.pth")
# model.get_policy()
model.plot_start_state('LARGE_ENV')
model.plot_rewards('LARGE_ENV')
# policy = model.learned_policy

model = torch.load('dqn_large.pth')
v = np.zeros(n*n)
policy = np.zeros(n*n)

for i in range(n*n):
        q_values = model(torch.FloatTensor([i])).view(-1)
        q_values = q_values.detach().numpy()
        action = np.argmax(q_values)
        policy[i] = action
        v[i] = np.max(q_values)


heat_map(v, policy, n)

# print("Simulating learned policy")
# obs, info = env.reset()
# rgb_image = env.render()
# plt.imshow(rgb_image)
# plt.savefig('fig.png')
# time.sleep(2)
# terminated = False
# truncated = False
# while not (terminated or truncated):
#         q_values = model(torch.FloatTensor([obs])).view(-1)
#         q_values = q_values.detach().numpy()
#         action = np.argmax(q_values)
#         print(action)
#         obs, reward, terminated, truncated, info = env.step(action)  
#         time.sleep(1)
#         rgb_image = env.render()
#         plt.imshow(rgb_image)
#         plt.savefig('fig.png')
#         time.sleep(1)

# observation, info = SMALL_ENV.reset()
# print(observation)
# print(info)
# SMALL_ENV.render()
# print(rgb_image.shape)
# # plt.plot([1,2,3],[2,4,6])
# plt.imshow(rgb_image)
# # plt.axis('off')  # Turn off axis
# plt.savefig('fig.png')
# # time.sleep(5)











