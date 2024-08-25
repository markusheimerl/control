import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from env import create_state, step, create_random_action

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, epsilon=0.2, value_coef=0.5, entropy_coef=0.01):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.value = ValueNetwork(state_dim)
        self.optimizer = optim.Adam(list(self.policy.parameters()) + list(self.value.parameters()), lr=lr)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.action_dim = action_dim

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = self.policy(state)
        return action.squeeze(0).numpy()

    def compute_returns(self, rewards, dones):
        returns = []
        R = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            R = reward + self.gamma * R * (1 - done)
            returns.insert(0, R)
        return returns

    def update(self, states, actions, old_log_probs, returns, advantages):
        states = torch.stack(states)
        actions = torch.stack(actions)
        old_log_probs = torch.stack(old_log_probs)
        returns = torch.tensor(returns).unsqueeze(1)
        advantages = torch.tensor(advantages).unsqueeze(1)

        for _ in range(10):  # PPO update iterations
            new_actions = self.policy(states)
            new_log_probs = torch.log(new_actions + 1e-8).sum(-1, keepdim=True)
            value_pred = self.value(states)

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = nn.MSELoss()(value_pred, returns)
            entropy = -(new_log_probs * torch.exp(new_log_probs)).mean()

            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

def train(num_episodes=1000, max_steps=1000):
    state_dim = 22  # Updated to match the flattened state size
    action_dim = 4
    ppo = PPO(state_dim, action_dim)

    for episode in range(num_episodes):
        state = create_state()
        states, actions, rewards, log_probs, dones = [], [], [], [], []

        for t in range(max_steps):
            state_flat = np.concatenate([
                state['omega'],
                state['angular_velocity_B'],
                state['linear_velocity_W'],
                state['linear_position_W'],
                state['R_W_B'].flatten()
            ])
            state_tensor = torch.FloatTensor(state_flat)

            action = ppo.get_action(state_tensor)
            action_tensor = torch.FloatTensor(action)
            log_prob = torch.log(action_tensor + 1e-8).sum()  # Sum log probs across action dimensions

            next_state, reward = step(state, action)
            done = False  # You need to implement a termination condition

            states.append(state_tensor)
            actions.append(action_tensor)
            rewards.append(reward)
            log_probs.append(log_prob)
            dones.append(done)

            state = next_state

            if done:
                break

        returns = ppo.compute_returns(rewards, dones)
        advantages = [ret - ppo.value(s.unsqueeze(0)).item() for ret, s in zip(returns, states)]

        ppo.update(states, actions, log_probs, returns, advantages)

        if episode % 10 == 0:
            print(f"Episode {episode}, Average Reward: {sum(rewards)/len(rewards)}")

if __name__ == "__main__":
    train()