- [Assignment Description](https://lily-molybdenum-65d.notion.site/Assignment-IV-Learning-Agent-Behaviours-595717cea8cf40089be477e9f703fbd1)
- [Part I: Imitation Learning Implementation](./agents/mujoco_agents.py#L99)
- [Part II: Policy Gradients Implementation](./agents/mujoco_agents.py#L246)
- [Part III: Stable Baselines3](./agents/mujoco_agents.py#L366)
- [Report](./report.txt)

## Environment Setup

```bash
# Instruction to set up on Linux

cd A4

#Install rl_env for Part I and II
conda env create -f rl_env_lin.yml
conda activate rl_env
pip install -e .
```

```bash
#Install sb3 env from sb3_env_lin.yml
conda env create -f sb3_env_lin.yml
conda activate sb3
pip install -e .

#Install MujoCo
#1) Download the tar file from the link https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz

#2)extract the file 
tar -xzf mujoco210-linux-x86_64.tar.gz

#3)Move the mujoco210 into ~/.mujoco/
mkdir ~/.mujoco
mv ./mujoco210 ~/.mujoco/

#4) In case of Error: Export the paths to LD_LIBRARY_PATH only if you face errors
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<path-to-mujoco210>/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```

## Training
- The training for part I and II  can be initiated from the A4 folder as follows:
```
python scripts/train_agent.py --env_name <ENV> --exp_name <ALGO>  [optional tags]
```
- The training for part III can be initiated from the A4 folder as:
```
python scripts/train_sb.py --env_name <ENV> [optional tags]
```