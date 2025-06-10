import gymnasium as gym
import torch
import torch.nn.functional as F  # Corrected import
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
import rl_utils

class PolicyNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1) # dim=1 is correct for (batch_size, action_dim) output

class REINFORCE:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, device):
        self.policy_net = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        # Convert single state numpy array to a PyTorch tensor with batch dimension
        state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        probs = self.policy_net(state_tensor)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
            rewards = transition_dict['rewards'] 
            states = transition_dict['states']   
            actions = transition_dict['actions']

            # Calculate discounted returns (G_t)
            G = 0
            returns = []
            for r in reversed(rewards):
                G = self.gamma * G + r
                returns.insert(0, G)

            # Convert lists to PyTorch tensors
            returns = torch.tensor(returns, dtype=torch.float, device=self.device)
            returns = returns.view(-1, 1)   # <— 变成 (N,1)

            states_tensor = torch.tensor(np.array(states), dtype=torch.float).to(self.device)
            actions_tensor = torch.tensor(np.array(actions), dtype=torch.long).view(-1, 1).to(self.device)

            # --- FIX: Standardize the returns to reduce variance ---
            # Add a small epsilon to avoid division by zero
            returns = (returns - returns.mean()) / (returns.std() + 1e-9) 
            
            # Calculate log probabilities for the actions taken
            log_probs = torch.log(self.policy_net(states_tensor).gather(1, actions_tensor))

            # Calculate loss (policy gradient loss)
            # The returns are now standardized advantages
            loss = -torch.mean(log_probs * returns)

            # Perform optimization step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

# --- Training Loop ---
learning_rate = 5e-4
num_episodes = 1000
hidden_dim = 128
gamma = 0.98
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env_name = "CartPole-v0"
env = gym.make(env_name) # Ensure you have gymnasium installed

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = REINFORCE(state_dim, hidden_dim, action_dim, learning_rate, gamma, device)

return_list = []
for i in range(10): # This outer loop is just for tqdm iterations, total num_episodes is 1000
    with tqdm(total=int(num_episodes / 10), desc="Iteration %d" % i) as pbar:
        for i_episode in range(int(num_episodes / 10)): # This is actually num_episodes / 10 episodes
            episode_return = 0
            transition_dict = {
                "states": [],
                "actions": [],
                "rewards": [],
            }
            
            # --- FIX 1: Correctly unpack env.reset() output ---
            state, info = env.reset() 
            terminated = False # New Gymnasium API: terminated and truncated
            truncated = False

            # --- FIX 2: Correct loop condition for Gymnasium ---
            while not terminated and not truncated:
                action = agent.take_action(state)
                # --- FIX 3: Correctly unpack env.step() output ---
                next_state, reward, terminated, truncated, info = env.step(action) 
                
                transition_dict["states"].append(state)
                transition_dict["actions"].append(action)
                transition_dict["rewards"].append(reward) # Rewards are collected for G_t calculation
                
                state = next_state
                episode_return += reward
            
            return_list.append(episode_return)
            agent.update(transition_dict) # Update policy after each episode

            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('REINFORCE on {}'.format(env_name))
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('REINFORCE on {}'.format(env_name))
plt.show()