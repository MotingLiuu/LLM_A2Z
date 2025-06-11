import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename="actor-critic.log", encoding="utf-8", level=logging.DEBUG, filemode="w")

class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)
    
class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
class Actor_Critic:
    def __init__(self, state_dim, hidden_dim, action_dim, policy_learning_rate, value_learning_rate, gamma, device):
        self.policy_net = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.value_net = ValueNet(state_dim, hidden_dim).to(device)
        self.gamma = gamma
        self.policy_learning_rate = policy_learning_rate
        self.value_learning_rate = value_learning_rate
        self.device = device
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.policy_learning_rate)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=self.value_learning_rate)
    
    def take_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        probs = self.policy_net(state_tensor)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()
    
    def update(self, transition_dict):
        state = transition_dict["state"]
        action = transition_dict["action"]
        reward = transition_dict["reward"]
        next_state = transition_dict["next_state"]
        truancated = transition_dict["truncated"]
        terminated = transition_dict["terminated"]
        mask = 1.0 if not terminated else 0.0
        
        state_tensor = torch.tensor(np.array(state), dtype=torch.float).to(self.device)
        action_tensor = torch.tensor(np.array(action), dtype=torch.long).unsqueeze(0).to(self.device)
        reward_tensor = torch.tensor(np.array(reward), dtype=torch.float).to(self.device)
        next_state_tensor = torch.tensor(np.array(next_state), dtype=torch.float).to(self.device)

        value = self.value_net(state_tensor)
        td_target = reward_tensor + self.gamma * self.value_net(next_state_tensor) * mask
        
        log_prob = torch.log(self.policy_net(state_tensor).gather(1, action_tensor))
        td_delta = td_target - value
        
        policy_loss = -log_prob * td_delta.detach()
        value_loss = -value * td_delta.detach()
        
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        policy_loss.backward()
        value_loss.backward()
        self.policy_optimizer.step()
        self.value_optimizer.step()
      
policy_learning_rate = 1e-4
value_learning_rate = 1e-3
num_episodes = 1000
hidden_dim = 128
gamma = 0.98
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env_name = "CartPole-v1"
env = gym.make(env_name)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = Actor_Critic(state_dim, hidden_dim, action_dim, policy_learning_rate, value_learning_rate, gamma, device)

return_list = []
for i in range(10):
    for i_episode in tqdm(range(int(num_episodes / 10))):
        episode_return = 0
        state, info = env.reset()
        terminated = False
        truncated = False
        while not terminated and not truncated:
            transition_dict = {
                "state": [],
                "action": [],
                "reward": [],
                "next_state": [],
                "terminated": False,
                "truncated": False,
            }
            transition_dict["state"].append(state)
            action = agent.take_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            transition_dict["action"].append(action)
            transition_dict["reward"].append(reward)
            transition_dict["next_state"].append(next_state)
            transition_dict["terminated"] = terminated
            transition_dict["truncated"] = truncated
            
            episode_return += reward
            state = next_state
            agent.update(transition_dict)
        if i_episode % 10 == 0:
            logger.info(f"The {i * num_episodes/ 10 + i_episode}th reward is {episode_return}")
        return_list.append(episode_return)
        
plt.figure(figsize=(12, 6))
plt.plot(range(len(return_list)), return_list)
plt.grid(True)
plt.show()