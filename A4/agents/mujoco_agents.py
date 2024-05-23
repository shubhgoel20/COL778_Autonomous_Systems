import os
import itertools
import torch
import random
from torch import nn
from torch import optim
import numpy as np
from tqdm import tqdm
from torch.distributions import Categorical, Normal

from utils.replay_buffer import ReplayBuffer
import utils.utils as utils
from agents.base_agent import BaseAgent
import utils.pytorch_util as ptu
from policies.experts import load_expert_policy
import cv2
import time

seed_value = 42  # You can use any integer value
np.random.seed(seed_value)

def sample_trajectory_expert(
    env, policy, max_length: int, render: bool = False
):
    """Sample a rollout in the environment from a policy."""
    ob = env.reset()
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0

    while True:
        # render an image
        if render:
            # breakpoint()
            if hasattr(env, "sim"):
                img = env.sim.render(camera_name="track", height=500, width=500)[::-1]
            else:
                img = env.render(mode='single_rgb_array')
            image_obs.append(
                cv2.resize(img, dsize=(250, 250), interpolation=cv2.INTER_CUBIC)
            )

        #use the most recent ob to decide what to do
        in_ = torch.FloatTensor(ob)
        in_ = in_.unsqueeze(0)
        in_ = in_.to(ptu.device)
        # print(in_.device)
        ac = policy.get_action(in_)
        ac = ac[0]

        # Take that action and get reward and next ob
        next_ob, rew, terminated, _ = env.step(ac)
        
        # Rollout can end due to done, or due to max_path_length
        steps += 1
        rollout_done = (terminated) or (steps >= max_length) # HINT: this is either 0 or 1

        # record result of taking that action
        obs.append(ob)
        acs.append(ac)
        rewards.append(rew)
        next_obs.append(next_ob)
        terminals.append(rollout_done)

        ob = next_ob  # jump to next timestep

        # end the rollout if the rollout ended
        if rollout_done:
            break

    return {
        "observation": np.array(obs, dtype=np.float32),
        "image_obs": np.array(image_obs, dtype=np.uint8),
        "reward": np.array(rewards, dtype=np.float32),
        "action": np.array(acs, dtype=np.float32),
        "next_observation": np.array(next_obs, dtype=np.float32),
        "terminal": np.array(terminals, dtype=np.float32),
    }


def sample_trajectories_expert(
    env,
    policy,
    min_timesteps_per_batch: int,
    max_length: int,
    render: bool = False,
):
    """Collect rollouts using policy until we have collected min_timesteps_per_batch steps."""
    timesteps_this_batch = 0
    trajs = []
    while timesteps_this_batch < min_timesteps_per_batch:
        # collect rollout
        traj = sample_trajectory_expert(env, policy, max_length, render)
        trajs.append(traj)

        # count steps
        timesteps_this_batch += utils.get_traj_length(traj)
    return trajs, timesteps_this_batch

class ImitationAgent(BaseAgent):
    '''
    Please implement an Imitation Learning agent. Read scripts/train_agent.py to see how the class is used. 
    
    
    Note: 1) You may explore the files in utils to see what helper functions are available for you.
          2)You can add extra functions or modify existing functions. Dont modify the function signature of __init__ and train_iteration.  
          3) The hyperparameters dictionary contains all the parameters you have set for your agent. You can find the details of parameters in config.py.  
    
    Usage of Expert policy:
        Use self.expert_policy.get_action(observation:torch.Tensor) to get expert action for any given observation. 
        Expert policy expects a CPU tensors. If your input observations are in GPU, then 
        You can explore policies/experts.py to see how this function is implemented.
    '''

    def __init__(self, observation_dim:int, action_dim:int, args = None, discrete:bool = False, **hyperparameters ):
        super().__init__()
        self.hyperparameters = hyperparameters
        self.action_dim  = action_dim
        self.observation_dim = observation_dim
        self.is_action_discrete = discrete
        self.args = args
        self.replay_buffer = ReplayBuffer(self.hyperparameters['buffer_size']) #you can set the max size of replay buffer if you want

        #Hyperparameters
        self.n_layers = self.hyperparameters['n_layers']
        self.hidden_layers = self.hyperparameters['hidden_size']
        self.beta = self.hyperparameters['beta']
        self.max_step_per_traj = self.hyperparameters['max_step_per_traj']
        self.min_steps_per_batch = self.hyperparameters['min_steps_per_batch']
        self.num_epochs = self.hyperparameters['num_epochs']
        self.batch_size = self.hyperparameters['batch_size']
        self.lr = self.hyperparameters['lr']
        self.min_beta = self.hyperparameters['min_beta']
        self.decay = self.hyperparameters['decay']
        self.env = self.hyperparameters['env']
        self.save_freq = self.hyperparameters['save_freq']

        self.save_path = "./best_models"
        os.makedirs(self.save_path, exist_ok=True)  # exist_ok=True allows the directory to exist without throwing an error
        print(f"Directory '{self.save_path}' was created successfully or already exists.")
        

        #initialize your model and optimizer and other variables you may need
        self.policy = ptu.build_mlp(observation_dim,action_dim,self.n_layers, self.hidden_layers, output_activation='tanh')
        self.optimizer = optim.Adam(self.policy.parameters(), lr= self.lr)

        

    def forward(self, observation: torch.FloatTensor):
        #*********YOUR CODE HERE******************
        action = self.policy(observation)
        return action


    @torch.no_grad()
    def get_action(self, observation: torch.FloatTensor):
        #*********YOUR CODE HERE******************
        action  = self.forward(observation)
        # action = action.detach().cpu().numpy()
        return action 

    
    
    def update(self, observations, actions):
        #*********YOUR CODE HERE******************
        loss_fn = nn.MSELoss()
        predictions = self.forward(observations)
        targets = actions
        loss = loss_fn(predictions, targets)
        return loss
    


    def train_iteration(self, env, envsteps_so_far, render=False, itr_num=None, **kwargs):
        if not hasattr(self, "expert_policy"):
            self.expert_policy, initial_expert_data = load_expert_policy(env, self.args.env_name)
            self.replay_buffer.add_rollouts(initial_expert_data)
            
            #to sample from replay buffer use self.replay_buffer.sample_batch(batch_size, required = <list of required keys>)
            # for example: sample = self.replay_buffer.sample_batch(32)
        
        #*********YOUR CODE HERE******************
        # 1)Select the Policy
        random_num = np.random.random()
        use = 'agent'
        print(random_num)
        if (random_num < self.beta):
            use = 'expert'

        # beta decay logic
        if itr_num > 10:
            if self.beta > self.min_beta:
                self.beta*=self.decay

        # 2)Sample T-step trajectory
        if use == 'expert':
            trajs, num_time_steps = sample_trajectories_expert(env,self.expert_policy, self.min_steps_per_batch, self.max_step_per_traj)
        else:
            trajs, num_time_steps = utils.sample_trajectories(env,self, self.min_steps_per_batch, self.max_step_per_traj)
        print(use)
        # 3)Get expert actions on the visited observation
        for i,traj in enumerate(trajs):
            for j, obs in enumerate(traj['observation']):
                trajs[i]['action'][j] = self.expert_policy.get_action(torch.FloatTensor(obs).to(ptu.device))

        # 4)Update replay_buffer
        self.replay_buffer.add_rollouts(trajs)
        
        # 5)Train the policy
        episode_loss = 0.0
        
        samples = []
        #was 50 for hopper
        for i in range(100):
            samples.append(self.replay_buffer.sample_batch(self.batch_size))
        
        for epoch in range(1, self.num_epochs+1):
            epoch_loss = 0.0
            for sample in samples:
            # sample = self.replay_buffer.sample_batch(self.batch_size)
            # print(sample.keys())
                loss = self.update(torch.FloatTensor(sample['obs']).to(ptu.device), torch.FloatTensor(sample['acs']).to(ptu.device))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss+=loss.item()
            epoch_loss/=len(samples)
            episode_loss+=epoch_loss
            if epoch%1 == 0:
                print(f"Epoch: {epoch}, loss: {epoch_loss}")
        episode_loss/=self.num_epochs

        if itr_num%self.save_freq == 0:
            torch.save(self.state_dict(), os.path.join(self.save_path,"imitation_"+self.env+".pth"))
        
        # print(self.replay_buffer.__len__())
        return {'episode_loss': episode_loss, 'trajectories': trajs, 'current_train_envsteps': num_time_steps} #you can return more metadata if you want to









class RLAgent(BaseAgent):

    '''
    Please implement an policy gradient agent. Read scripts/train_agent.py to see how the class is used. 
    
    
    Note: Please read the note (1), (2), (3) in ImitationAgent class. 
    '''

    def __init__(self, observation_dim:int, action_dim:int, args = None, discrete:bool = False, **hyperparameters ):
        super().__init__()
        self.hyperparameters = hyperparameters
        self.action_dim  = action_dim
        self.observation_dim = observation_dim
        self.is_action_discrete = discrete
        self.args = args
        self.discrete = discrete
        #initialize your model and optimizer and other variables you may need

        # Network setup
        self.n_layers = self.hyperparameters['n_layers']
        self.hidden_size = self.hyperparameters['hidden_size']
        self.policy_network =  ptu.build_mlp(observation_dim,action_dim,self.n_layers, self.hidden_size, output_activation='tanh')
        if not self.discrete:
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        self.optimizer = optim.Adam(list(self.policy_network.parameters()) + [self.log_std], lr=self.hyperparameters.get('lr'))
        folder_path = "./best_models"
        os.makedirs(folder_path, exist_ok=True)  # exist_ok=True allows the directory to exist without throwing an error
        print(f"Directory '{folder_path}' was created successfully or already exists.")
        self.best_training_reward = -1e8

    def forward(self, observation: torch.FloatTensor):
        #*********YOUR CODE HERE******************
        return self.policy_network(observation)

    @torch.no_grad()
    def get_action(self, observation: torch.FloatTensor):
        #*********YOUR CODE HERE******************
        logits = self.policy_network(observation)
        if self.discrete:
            distribution = Categorical(logits=logits)
        else:
            std = torch.exp(self.log_std)
            distribution = Normal(logits, std)
        action = distribution.sample()
        # action = torch.clamp(action, min=-1, max=1)
        return action



    def train_iteration(self, env, envsteps_so_far, render=False, itr_num=None, **kwargs):
        #*********YOUR CODE HERE******************
        self.train()

        # Sample multiple trajectories
        min_timesteps_per_batch = self.hyperparameters.get('min_timesteps_per_batch', 500)
        max_trajectory_length = self.hyperparameters.get('max_trajectory_length', 500)
        trajs, total_timesteps = utils.sample_trajectories(env, self.get_action, min_timesteps_per_batch, max_trajectory_length, render)


        loss = 0
        for traj in trajs:
            discounted_rewards = []
            G = 0
            for r in reversed(traj["reward"]):
                G = r + self.hyperparameters.get('gamma') * G
                discounted_rewards.insert(0, G)
            discounted_rewards = np.array(discounted_rewards)
            traj['discounted_rewards'] = discounted_rewards

        # baseline_reward = sum(traj['discounted_rewards'][0] for traj in trajs)/len(trajs)
        reward_mean = np.mean(np.concatenate([traj['discounted_rewards'].flatten() for traj in trajs]))
        # reward_std = np.std(np.concatenate([traj['discounted_rewards'].flatten() for traj in trajs]))

        training_reward = np.mean([traj["reward"].sum() for traj in trajs])
        if training_reward > self.best_training_reward:
            self.best_training_reward = training_reward
            torch.save(self.state_dict(), './best_models/rl_Hopper-v4.pth')
        for traj in trajs:
            # Convert lists to tensors
            observations_tensor = torch.tensor(traj['observation'], dtype=torch.float32, device=ptu.device) ## T x 11
            actions_tensor = torch.tensor(traj['action'], dtype=torch.float32, device=ptu.device) ## T x 3
            rewards_tensor = torch.tensor(traj['discounted_rewards']-reward_mean, dtype=torch.float32, device=ptu.device) ## T
            # print(observations_tensor.shape, actions_tensor.shape, rewards_tensor.shape) 

            if self.discrete:
                action_dists = self.forward(observations_tensor)
                dists = Categorical(logits=action_dists)
            else:
                action_dists = self.forward(observations_tensor)
                stds = torch.exp(self.log_std)
                dists = Normal(action_dists, stds)

            log_probs = dists.log_prob(actions_tensor).sum(axis=-1)

            # Policy Gradient update
            loss += -(log_probs * rewards_tensor).sum()
        
        loss /= len(trajs)
        self.optimizer.zero_grad()
        loss.backward()

        # clip_norm = 10  # Maximum norm of the gradients
        # torch.nn.utils.clip_grad_norm_(list(self.policy_network.parameters()) + [self.log_std], clip_norm)
        self.optimizer.step()

        return {
            'episode_loss': loss.item(),
            'trajectories': trajs,
            'current_train_envsteps':  total_timesteps
        }








class SBAgent(BaseAgent):
    def __init__(self, env_name, **hyperparameters):
        #implement your init function
        from stable_baselines3.common.env_util import make_vec_env
        import gymnasium as gym
        from stable_baselines3 import PPO, HerReplayBuffer, SAC, DDPG, TD3
        from stable_baselines3.common.callbacks import CheckpointCallback
        import panda_gym

        self.hyperparameters = hyperparameters
        self.env_name = env_name
        if self.env_name == "PandaPush-v3":
            n_sampled_goal = self.hyperparameters["n_sampled_goal"]
            self.env = gym.make(self.env_name) #initialize your environment. This variable will be used for evaluation. See train_sb.py
            self.model = SAC(
                            "MultiInputPolicy",
                            self.env,
                            replay_buffer_class=HerReplayBuffer,
                            replay_buffer_kwargs=dict(
                            n_sampled_goal=n_sampled_goal,
                            goal_selection_strategy="future",
                            ),
                            verbose=1,
                            buffer_size=self.hyperparameters["buffer_size"],
                            learning_rate=self.hyperparameters["learning_rate"],
                            gamma=self.hyperparameters["gamma"],
                            batch_size=self.hyperparameters["batch_size"],
                            tensorboard_log='../data/'
                        )
        else:
            self.env = make_vec_env(self.env_name, n_envs=1)
            self.env_temp = make_vec_env(self.env_name, n_envs=self.hyperparameters["n_envs"]) #initialize your environment. This variable will be used for evaluation. See train_sb.py
            self.model = TD3("MlpPolicy", self.env_temp, verbose=1, learning_rate=self.hyperparameters["learning_rate"], tensorboard_log='../data/')
        
        self.save_path = './best_models'
        os.makedirs(self.save_path, exist_ok=True)  # exist_ok=True allows the directory to exist without throwing an error
        print(f"Directory '{self.save_path}' was created successfully or already exists.")
    
    def learn(self):
        #implement your learn function. You should save the checkpoint periodically to <env_name>_sb3.zip
        from stable_baselines3.common.callbacks import CheckpointCallback
        start_time = time.time()
        if self.env_name == 'PandaPush-v3':
            max_time = 10.5 * 60 * 60 
        elif self.env_name == 'Ant-v4':
            max_time = 6.5 * 60 * 60
        class CustomCheckpointCallback(CheckpointCallback):
            def __init__(self, save_freq, save_path, name_prefix, save_replay_buffer=False, save_vecnormalize=False):
                super(CustomCheckpointCallback, self).__init__(save_freq, save_path, name_prefix, save_replay_buffer, save_vecnormalize)

            def on_step(self) -> bool:
                time_elapsed = time.time() - start_time
                if (self.n_calls % self.save_freq == 0) and (time_elapsed < max_time):
                    print(f"{time_elapsed:.2f}s")
                    path = os.path.join(self.save_path, f"{self.name_prefix}")
                    self.model.save(path)
                    if self.verbose > 1:
                        print(f"Saving model checkpoint to {path}")
                self.n_calls += 1
                return True
        checkpoint_callback = CustomCheckpointCallback(save_freq=1e4, save_path= self.save_path, name_prefix="sb_"+self.env_name)
        self.model.learn(total_timesteps=self.hyperparameters["total_timesteps"], callback = checkpoint_callback)
        # self.model.save(self.env_name + "_sb3.pth")
    
    def load_checkpoint(self, checkpoint_path):
        #implement your load checkpoint function
        from stable_baselines3 import PPO, SAC, DDPG, TD3
        if self.env_name == "PandaPush-v3":
            self.model = SAC.load(checkpoint_path, env=self.env)
        else:
            self.model = TD3.load(checkpoint_path)
        
    
    def get_action(self, observation):
        #implement your get action function
        if self.env_name == "PandaPush-v3":
            action, _ = self.model.predict(observation, deterministic=True)
        else:
            action, _ = self.model.predict(observation)
        return action, _
    