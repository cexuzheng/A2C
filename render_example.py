from a2c import A2C
import gym
import torch.optim as optim
import torch
from torch.distributions import Categorical

import os

# Load
LR = .01  # Learning rate
SEED = None  # Random seed for reproducibility
MAX_EPISODES = 10000  # Max number of episodes

agent = A2C(gym.make('CartPole-v1'), random_seed=SEED)

# Init optimizers
actor_optim = optim.Adam(agent.actor.parameters(), lr=LR)
critic_optim = optim.Adam(agent.critic.parameters(), lr=LR)
agent.actor.load_state_dict(torch.load('actor_state_dict'))
agent.critic.load_state_dict(torch.load('critic_state_dict'))

environment_name = "CartPole-v1" #Env
env = gym.make(environment_name,render_mode="human")

ppo_path = os.path.join('Cartpole_model')

obs, info = env.reset()
while True:
    action_logits = agent.actor( torch.from_numpy(obs).double() )
    action = Categorical(logits=action_logits).sample()
    obs, rewards,  dones, trunc, info = env.step(action.item())
    env.render()

env.close()