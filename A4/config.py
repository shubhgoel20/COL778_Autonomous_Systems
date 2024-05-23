'''
This file contains the configs used for Model creation and training. You need to give your best hyperparameters and the configs you used to get the best results for 
every environment and experiment.  These configs will be automatically loaded and used to create and train your model in our servers.
'''
#You can add extra keys or modify to the values of the existing keys in bottom level of the dictionary.
#DONT CHANGE THE OVERALL STRUCTURE OF THE DICTIONARY. 

configs = {
    

    'Hopper-v4': {
        "imitation":{
            #You can add or change the keys here
              "hyperparameters": {
                'hidden_size': 64,
                'n_layers': 3,
                'batch_size': 256, 
                'beta':0.8,
                'max_step_per_traj':1000,
                'min_steps_per_batch':5000,
                'num_epochs':10,
                'lr':1e-3,
                'min_beta':0.1,
                'decay': 0.9,
                'buffer_size': 25000,
                'env': 'Hopper-v4',
                'save_freq': 5,
            },
            "num_iteration": 170,

        },

        "RL":{
            #You can add or change the keys here
               "hyperparameters": {
                'hidden_size': 64,
                'n_layers': 3,
                'batch_size': 512, 
                'gamma': 0.999,
                'lr': 1e-4
                
            },
            "num_iteration": 50000,
        },
    },
    
    
    'Ant-v4': {
        "imitation":{
            #You can add or change the keys here
              "hyperparameters": {
                'hidden_size': 128,
                'n_layers': 3,
                'batch_size': 256, 
                'beta':0.8,
                'max_step_per_traj':3000,
                'min_steps_per_batch':5000,
                'num_epochs':10,
                'lr':1e-3,
                'min_beta':0.1,
                'decay': 0.85,
                'buffer_size': 25000,
                'env': "Ant-v4",
                'save_freq': 5,
            },
            "num_iteration": 151,
            "episode_len": 1500

        },
        
        "SB3":{
            #You can add or change the keys here
                "hyperparameters": {
                    "total_timesteps": 1e6,
                    "n_envs": 4,
                    "learning_rate": 1e-3,
                },            
        },
    },
    
    'PandaPush-v3': {
        "SB3":{
            #You can add or change the keys here
                "hyperparameters": {
                    "total_timesteps": 1e6,
                    "buffer_size": int(1e6),
                    "learning_rate": 3e-4,
                    "gamma": 0.99,
                    "batch_size": 512,
                    "n_sampled_goal": 4,
                },            
        },
    },

}