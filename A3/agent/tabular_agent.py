import gymnasium as gym
import car_gym
from utils import SMALL_ENV, LARGE_ENV
from utils import plot_rewards
import matplotlib.pyplot as plt
import time
import numpy as np

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
class QLearning:
        def __init__(self, env, alpha, gamma, eps, num_episodes):
                self.env = env
                self.alpha = alpha
                self.gamma = gamma
                self.eps = eps
                self.num_episodes = num_episodes
                self.learned_policy = np.zeros(env.observation_space.n)
                self.num_states = env.observation_space.n
                self.num_actions = env.action_space.n
                self.Qmat = np.zeros((self.num_states, self.num_actions))
                self.start_state_q_value = []

        def select_action(self, state, episode_no):
                # if episode_no < 1000:
                #         return np.random.choice(self.num_actions)

                random_num = np.random.random()

                if random_num < self.eps:
                        return np.random.choice(self.num_actions)
                else:
                        return np.argmax(self.Qmat[state])


        def simulate(self):
                for episode in range(self.num_episodes):
                        obs, info = self.env.reset()
                        print("Simulating episode {}".format(episode))
                        terminated = False
                        truncated = False
                        while not (terminated or truncated):
                                action = self.select_action(obs, episode)
                                next_obs, reward, terminated, truncated, info = self.env.step(action)
                                next_action = np.argmax(self.Qmat[next_obs])

                                if not (terminated):
                                        target = reward + self.gamma*self.Qmat[next_obs,next_action]
                                        self.Qmat[obs, action] = self.Qmat[obs, action] + self.alpha*(target - self.Qmat[obs, action])
                                else:
                                        target = reward
                                        self.Qmat[obs, action] = self.Qmat[obs, action] + self.alpha*(target - self.Qmat[obs, action])

                                obs = next_obs
                        self.start_state_q_value.append(np.max(self.Qmat[0]))
                        if episode > 25000:
                                if self.eps > 0.5: self.eps = 0.999*self.eps
        
        def get_policy(self):
                for state in range(self.num_states):
                        # self.learned_policy[state]= np.random.choice(np.where(self.Qmat[state]==np.max(self.Qmat[state]))[0])
                        self.learned_policy[state]= np.argmax(self.Qmat[state])

        def plot_start_state(self, env_type):
                plt.plot(self.start_state_q_value, marker = 'o', markersize = 3)
                plt.xlabel('Episode Number')
                plt.ylabel('Start State Estimated Value')
                plt.title(env_type)
                plt.grid(True, ls = '--')
                plt.savefig(env_type+".png")
                plt.close()


dict = {0:np.array([-1,0], dtype=float), 
        1 : np.array([0,1], dtype=float), 
        2 : np.array([1,0], dtype=float), 
        3 : np.array([0,-1], dtype=float)}

def heat_map(v,policy,n):
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
        plt.savefig('heat_map_small.png')
        plt.close()
        # Show plot
        # plt.show()

#TODO

env = SMALL_ENV
alpha = 0.15   #0.9 0.64 0.32 0.16 0.08
gamma = 0.99
eps = 1
num_episodes = 75000
n = 20

model = QLearning(env,alpha,gamma,eps,num_episodes)
model.simulate()
model.get_policy()
model.plot_start_state('SMALL_ENV')

policy = model.learned_policy
v = np.zeros(n*n)
Qmat = model.Qmat

for i in range(n*n):
        v[i] = np.max(Qmat[i])

heat_map(v, policy, n)


print("Simulating learned policy")
obs, info = env.reset()
rgb_image = env.render()
plt.imshow(rgb_image)
plt.savefig('fig.png')
time.sleep(2)
terminated = False
while not terminated:
        obs, reward, terminated, truncated, info = env.step(int(policy[obs]))
        print(reward)  
        time.sleep(1)
        rgb_image = env.render()
        plt.imshow(rgb_image)
        plt.savefig('fig.png')
        time.sleep(1)

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











