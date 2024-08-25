import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Constants
k_f = 0.0004905
k_m = 0.00004905
L = 0.25
l = L / np.sqrt(2)
I = np.array([0.01, 0.02, 0.01])
loc_I_mat = np.diag(I)
loc_I_mat_inv = np.linalg.inv(loc_I_mat)
g = 9.81
m = 0.5
dt = 0.01
omega_min = 30
omega_max = 70
omega_stable = 50

def create_random_rotation_matrix():
    x = np.random.uniform(-np.pi, np.pi)
    y = np.random.uniform(-np.pi, np.pi)
    z = np.random.uniform(-np.pi, np.pi)
    
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(x), -np.sin(x)],
                   [0, np.sin(x), np.cos(x)]])
    
    Ry = np.array([[np.cos(y), 0, np.sin(y)],
                   [0, 1, 0],
                   [-np.sin(y), 0, np.cos(y)]])
    
    Rz = np.array([[np.cos(z), -np.sin(z), 0],
                   [np.sin(z), np.cos(z), 0],
                   [0, 0, 1]])
    
    return Rz @ Ry @ Rx

def create_state():
    return {
        'omega': np.random.uniform(omega_min, omega_max, 4).astype(np.float32),
        'angular_velocity_B': np.random.uniform(-1, 1, 3).astype(np.float32),
        'linear_velocity_W': np.random.uniform(-1, 1, 3).astype(np.float32),
        'linear_position_W': np.random.uniform(-5, 5, 3).astype(np.float32),
        'R_W_B': create_random_rotation_matrix().astype(np.float32)
    }

def calculate_reward(state):
    target_omega = np.array([omega_stable] * 4, dtype=np.float32)
    target_angular_velocity = np.zeros(3, dtype=np.float32)
    target_linear_velocity = np.zeros(3, dtype=np.float32)
    target_linear_position = np.zeros(3, dtype=np.float32)
    target_rotation = np.eye(3, dtype=np.float32)

    omega_error = np.linalg.norm(state['omega'] - target_omega)
    angular_velocity_error = np.linalg.norm(state['angular_velocity_B'] - target_angular_velocity)
    linear_velocity_error = np.linalg.norm(state['linear_velocity_W'] - target_linear_velocity)
    position_error = np.linalg.norm(state['linear_position_W'] - target_linear_position)
    rotation_error = np.linalg.norm(state['R_W_B'] - target_rotation)

    reward = -(omega_error + angular_velocity_error + linear_velocity_error + position_error + rotation_error)
    return np.clip(reward, -100, 0)  # Clip reward to avoid extreme values

def step(state, action):
    omega = np.clip(action, omega_min, omega_max)
    angular_velocity_B = state['angular_velocity_B']
    linear_velocity_W = state['linear_velocity_W']
    linear_position_W = state['linear_position_W']
    R_W_B = state['R_W_B']

    F = k_f * omega * np.abs(omega)
    M = k_m * omega * np.abs(omega)

    f_B_thrust = np.array([0, np.sum(F), 0], dtype=np.float32)

    tau_B_drag = np.array([0, M[0] - M[1] + M[2] - M[3], 0], dtype=np.float32)
    tau_B_thrust = np.array([
        L * (F[0] - F[1] - F[2] + F[3]),
        0,
        L * (-F[0] - F[1] + F[2] + F[3])
    ], dtype=np.float32)
    tau_B = tau_B_drag + tau_B_thrust

    linear_acceleration_W = np.array([0, -g, 0], dtype=np.float32) + R_W_B @ f_B_thrust / m
    linear_acceleration_W = np.clip(linear_acceleration_W, -100, 100)  # Clip acceleration to avoid extreme values

    angular_acceleration_B = np.linalg.inv(loc_I_mat) @ (
        -np.cross(angular_velocity_B, loc_I_mat @ angular_velocity_B) + tau_B
    )
    angular_acceleration_B = np.clip(angular_acceleration_B, -100, 100)  # Clip angular acceleration

    linear_velocity_W += dt * linear_acceleration_W
    linear_position_W += dt * linear_velocity_W
    angular_velocity_B += dt * angular_acceleration_B

    R_W_B += dt * R_W_B @ np.array([
        [0, -angular_velocity_B[2], angular_velocity_B[1]],
        [angular_velocity_B[2], 0, -angular_velocity_B[0]],
        [-angular_velocity_B[1], angular_velocity_B[0], 0]
    ], dtype=np.float32)

    # Normalize R_W_B to ensure it remains a valid rotation matrix
    U, _, Vt = np.linalg.svd(R_W_B)
    R_W_B = U @ Vt

    new_state = {
        'omega': omega,
        'angular_velocity_B': angular_velocity_B,
        'linear_velocity_W': linear_velocity_W,
        'linear_position_W': linear_position_W,
        'R_W_B': R_W_B
    }

    reward = calculate_reward(new_state)

    return new_state, reward

def is_done(state):
    position_threshold = 0.1
    velocity_threshold = 0.1
    angular_velocity_threshold = 0.1
    rotation_threshold = 0.1

    position_stable = np.linalg.norm(state['linear_position_W']) < position_threshold
    velocity_stable = np.linalg.norm(state['linear_velocity_W']) < velocity_threshold
    angular_velocity_stable = np.linalg.norm(state['angular_velocity_B']) < angular_velocity_threshold
    rotation_stable = np.linalg.norm(state['R_W_B'] - np.eye(3)) < rotation_threshold

    return position_stable and velocity_stable and angular_velocity_stable and rotation_stable

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x)) * (omega_max - omega_min) + omega_min

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, epsilon=0.2, value_coef=0.5, entropy_coef=0.01):
        self.policy = PolicyNetwork(state_dim, action_dim).to(device)
        self.value = ValueNetwork(state_dim).to(device)
        self.optimizer = optim.Adam(list(self.policy.parameters()) + list(self.value.parameters()), lr=lr)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.action_dim = action_dim

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = self.policy(state)
        return action.squeeze(0).cpu().numpy()

    def compute_returns(self, rewards, dones):
        returns = []
        R = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            R = reward + self.gamma * R * (1 - done)
            returns.insert(0, R)
        return returns

    def update(self, states, actions, old_log_probs, returns, advantages):
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.FloatTensor(np.array(actions)).to(device)
        old_log_probs = torch.FloatTensor(old_log_probs).unsqueeze(1).to(device)
        returns = torch.FloatTensor(returns).unsqueeze(1).to(device)
        advantages = torch.FloatTensor(advantages).unsqueeze(1).to(device)

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

            action = ppo.get_action(state_flat)
            log_prob = np.sum(np.log(action + 1e-8))

            next_state, reward = step(state, action)
            done = is_done(next_state)

            states.append(state_flat)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            dones.append(done)

            state = next_state

            if done:
                break

        returns = ppo.compute_returns(rewards, dones)
        advantages = [ret - ppo.value(torch.FloatTensor(s).unsqueeze(0).to(device)).item() for ret, s in zip(returns, states)]

        ppo.update(states, actions, log_probs, returns, advantages)

        if episode % 10 == 0:
            print(f"Episode {episode}, Average Reward: {sum(rewards)/len(rewards):.2f}, Steps: {len(rewards)}")

if __name__ == "__main__":
    train()