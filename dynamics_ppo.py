import numpy as np
import math
import random
import time

# Vector and matrix operations
def crossVec3f(v1, v2):
    return [
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0]
    ]

def multScalVec3f(s, v):
    return [v[0] * s, v[1] * s, v[2] * s]

def addVec3f(v1, v2):
    return [v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]]

def multMat3f(a, b):
    return [
        a[0] * b[0] + a[1] * b[3] + a[2] * b[6],
        a[0] * b[1] + a[1] * b[4] + a[2] * b[7],
        a[0] * b[2] + a[1] * b[5] + a[2] * b[8],
        a[3] * b[0] + a[4] * b[3] + a[5] * b[6],
        a[3] * b[1] + a[4] * b[4] + a[5] * b[7],
        a[3] * b[2] + a[4] * b[5] + a[5] * b[8],
        a[6] * b[0] + a[7] * b[3] + a[8] * b[6],
        a[6] * b[1] + a[7] * b[4] + a[8] * b[7],
        a[6] * b[2] + a[7] * b[5] + a[8] * b[8]
    ]

def multMatVec3f(m, v):
    return [
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
        m[6] * v[0] + m[7] * v[1] + m[8] * v[2]
    ]

def vecToDiagMat3f(v):
    return [
        v[0], 0.0, 0.0,
        0.0, v[1], 0.0,
        0.0, 0.0, v[2]
    ]

def invMat3f(m):
    det = (
        m[0] * (m[4] * m[8] - m[7] * m[5]) -
        m[1] * (m[3] * m[8] - m[5] * m[6]) +
        m[2] * (m[3] * m[7] - m[4] * m[6])
    )

    if det == 0:
        raise ValueError("Matrix is not invertible")

    invDet = 1.0 / det

    return [
        invDet * (m[4] * m[8] - m[7] * m[5]),
        invDet * (m[2] * m[7] - m[1] * m[8]),
        invDet * (m[1] * m[5] - m[2] * m[4]),
        invDet * (m[5] * m[6] - m[3] * m[8]),
        invDet * (m[0] * m[8] - m[2] * m[6]),
        invDet * (m[3] * m[2] - m[0] * m[5]),
        invDet * (m[3] * m[7] - m[6] * m[4]),
        invDet * (m[6] * m[1] - m[0] * m[7]),
        invDet * (m[0] * m[4] - m[3] * m[1])
    ]

def xRotMat3f(rads):
    s = math.sin(rads)
    c = math.cos(rads)
    return [
        1.0, 0.0, 0.0,
        0.0, c, -s,
        0.0, s, c
    ]

def yRotMat3f(rads):
    s = math.sin(rads)
    c = math.cos(rads)
    return [
        c, 0.0, s,
        0.0, 1.0, 0.0,
        -s, 0.0, c
    ]

def zRotMat3f(rads):
    s = math.sin(rads)
    c = math.cos(rads)
    return [
        c, -s, 0.0,
        s, c, 0.0,
        0.0, 0.0, 1.0
    ]

def so3hat(v):
    return [
        0.0, -v[2], v[1],
        v[2], 0.0, -v[0],
        -v[1], v[0], 0.0,
    ]

def addMat3f(a, b):
    return [
        a[0] + b[0], a[1] + b[1], a[2] + b[2],
        a[3] + b[3], a[4] + b[4], a[5] + b[5],
        a[6] + b[6], a[7] + b[7], a[8] + b[8]
    ]

def multScalMat3f(s, m):
    return [
        s * m[0], s * m[1], s * m[2],
        s * m[3], s * m[4], s * m[5],
        s * m[6], s * m[7], s * m[8]
    ]

# Constants
k_f = 0.0004905
k_m = 0.00004905
L = 0.25
l = L / math.sqrt(2)
I = [0.01, 0.02, 0.01]
loc_I_mat = vecToDiagMat3f(I)
loc_I_mat_inv = invMat3f(loc_I_mat)
g = 9.81
m = 0.5
dt = 0.01
omega_min = 30
omega_max = 70
omega_stable = 50

# IMU noise parameters
accel_noise_std = 0.05  # m/s^2
gyro_noise_std = 0.01  # rad/s

# Simulation parameters
max_iterations = 1000  # Run for 1000 iterations per simulation
num_simulations = 1000  # Run 1000 simulations

# Limits for random sampling
angular_velocity_max = 1.0  # rad/s
linear_velocity_max = 2.0  # m/s

def add_noise(value, std_dev):
    return value + random.gauss(0, std_dev)

def random_vector(max_value):
    return [random.uniform(-max_value, max_value) for _ in range(3)]

# PPO Hyperparameters
LEARNING_RATE = 0.0003
EPOCHS = 10
EPSILON = 0.2
GAMMA = 0.99
LAMBDA = 0.95

class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

    def add(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

class Actor:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = np.random.randn(input_dim, 64) / np.sqrt(input_dim)
        self.fc2 = np.random.randn(64, 64) / np.sqrt(64)
        self.fc3 = np.random.randn(64, output_dim) / np.sqrt(64)
        self.log_std = np.zeros(output_dim)

    def forward(self, x):
        x = np.tanh(np.dot(x, self.fc1))
        x = np.tanh(np.dot(x, self.fc2))
        x = np.tanh(np.dot(x, self.fc3))
        return x

    def sample_action(self, state):
        mean = self.forward(state)
        std = np.exp(self.log_std)
        action = mean + std * np.random.randn(self.output_dim)
        log_prob = -0.5 * np.sum(np.log(2 * np.pi * std**2) + (action - mean)**2 / (2 * std**2))
        return action, log_prob

class Critic:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.fc1 = np.random.randn(input_dim, 64) / np.sqrt(input_dim)
        self.fc2 = np.random.randn(64, 64) / np.sqrt(64)
        self.fc3 = np.random.randn(64, 1) / np.sqrt(64)

    def forward(self, x):
        x = np.tanh(np.dot(x, self.fc1))
        x = np.tanh(np.dot(x, self.fc2))
        x = np.dot(x, self.fc3)
        return x

class PPO:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.memory = PPOMemory()

    def choose_action(self, state):
        action, log_prob = self.actor.sample_action(state)
        value = self.critic.forward(state)
        return action, log_prob, value

    def update(self):
        for _ in range(EPOCHS):
            states = np.array(self.memory.states)
            actions = np.array(self.memory.actions)
            rewards = np.array(self.memory.rewards)
            values = np.array(self.memory.values)
            log_probs = np.array(self.memory.log_probs)
            dones = np.array(self.memory.dones)

            advantages = self.compute_gae(rewards, values, dones)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            for state, action, old_log_prob, advantage in zip(states, actions, log_probs, advantages):
                new_action, new_log_prob = self.actor.sample_action(state)
                ratio = np.exp(new_log_prob - old_log_prob)
                surr1 = ratio * advantage
                surr2 = np.clip(ratio, 1 - EPSILON, 1 + EPSILON) * advantage
                actor_loss = -np.minimum(surr1, surr2)

                value = self.critic.forward(state)
                critic_loss = (advantage + value - self.critic.forward(state))**2

                total_loss = actor_loss + 0.5 * critic_loss

                # Update actor and critic (simplified, without proper backpropagation)
                self.actor.fc1 -= LEARNING_RATE * np.outer(state, total_loss)
                self.actor.fc2 -= LEARNING_RATE * np.outer(np.tanh(np.dot(state, self.actor.fc1)), total_loss)
                self.actor.fc3 -= LEARNING_RATE * np.outer(np.tanh(np.dot(np.tanh(np.dot(state, self.actor.fc1)), self.actor.fc2)), total_loss)
                self.actor.log_std -= LEARNING_RATE * total_loss

                self.critic.fc1 -= LEARNING_RATE * np.outer(state, critic_loss)
                self.critic.fc2 -= LEARNING_RATE * np.outer(np.tanh(np.dot(state, self.critic.fc1)), critic_loss)
                self.critic.fc3 -= LEARNING_RATE * np.outer(np.tanh(np.dot(np.tanh(np.dot(state, self.critic.fc1)), self.critic.fc2)), critic_loss)

        self.memory.clear()

    def compute_gae(self, rewards, values, dones):
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        last_value = values[-1]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + GAMMA * last_value * (1 - dones[t]) - values[t]
            advantages[t] = last_advantage = delta + GAMMA * LAMBDA * last_advantage * (1 - dones[t])
            last_value = values[t]
        return advantages

def run_simulation_with_ppo(ppo_agent):
    # Initialize state with random values
    angular_velocity_B = random_vector(angular_velocity_max)
    linear_velocity_W = random_vector(linear_velocity_max)
    linear_position_W = [0, 1, 0]
    R_W_B = multMat3f(multMat3f(xRotMat3f(0), yRotMat3f(0)), zRotMat3f(0))

    total_reward = 0
    done = False

    for iteration in range(max_iterations):
        # Get state (IMU readings)
        imu_accel_reading = [add_noise(acc, accel_noise_std) for acc in linear_velocity_W]
        imu_gyro_reading = [add_noise(ang_vel, gyro_noise_std) for ang_vel in angular_velocity_B]
        state = np.array(imu_accel_reading + imu_gyro_reading)

        # Choose action using PPO
        action, log_prob, value = ppo_agent.choose_action(state)
        omega_1, omega_2, omega_3, omega_4 = np.clip(action, omega_min, omega_max)

        # Calculate forces and moments
        F1 = k_f * omega_1 * abs(omega_1)
        F2 = k_f * omega_2 * abs(omega_2)
        F3 = k_f * omega_3 * abs(omega_3)
        F4 = k_f * omega_4 * abs(omega_4)

        M1 = k_m * omega_1 * abs(omega_1)
        M2 = k_m * omega_2 * abs(omega_2)
        M3 = k_m * omega_3 * abs(omega_3)
        M4 = k_m * omega_4 * abs(omega_4)

        # Thrust
        f_B_thrust = [0, F1 + F2 + F3 + F4, 0]

        # Torque
        tau_B_drag = [0, M1 - M2 + M3 - M4, 0]
        tau_B_thrust_1 = crossVec3f([-L, 0, L], [0, F1, 0])
        tau_B_thrust_2 = crossVec3f([L, 0, L], [0, F2, 0])
        tau_B_thrust_3 = crossVec3f([L, 0, -L], [0, F3, 0])
        tau_B_thrust_4 = crossVec3f([-L, 0, -L], [0, F4, 0])
        tau_B_thrust = addVec3f(tau_B_thrust_1, tau_B_thrust_2)
        tau_B_thrust = addVec3f(tau_B_thrust, tau_B_thrust_3)
        tau_B_thrust = addVec3f(tau_B_thrust, tau_B_thrust_4)
        tau_B = addVec3f(tau_B_drag, tau_B_thrust)

        # Accelerations
        linear_acceleration_W = addVec3f([0, -g, 0], multScalVec3f(1/m, multMatVec3f(R_W_B, f_B_thrust)))
        angular_acceleration_B = multMatVec3f(loc_I_mat_inv, 
                                              addVec3f(tau_B, 
                                                       crossVec3f(multScalVec3f(-1, angular_velocity_B), 
                                                                  multMatVec3f(loc_I_mat, angular_velocity_B))))

        # Advance state
        linear_velocity_W = addVec3f(linear_velocity_W, multScalVec3f(dt, linear_acceleration_W))
        linear_position_W = addVec3f(linear_position_W, multScalVec3f(dt, linear_velocity_W))
        angular_velocity_B = addVec3f(angular_velocity_B, multScalVec3f(dt, angular_acceleration_B))
        R_W_B = addMat3f(R_W_B, multScalMat3f(dt, multMat3f(R_W_B, so3hat(angular_velocity_B))))

        # Calculate reward
        reward = -np.sum(np.abs(angular_velocity_B)) - np.sum(np.abs(linear_velocity_W))
        total_reward += reward

        # Check if done
        if np.all(np.abs(angular_velocity_B) < 0.01) and np.all(np.abs(linear_velocity_W) < 0.01):
            done = True

        # Store experience
        next_state = np.array(imu_accel_reading + imu_gyro_reading)
        ppo_agent.memory.add(state, action, reward, value, log_prob, done)

        if done or iteration == max_iterations - 1:
            break

    return total_reward, iteration + 1

def train_ppo():
    state_dim = 6  # 3 for accelerometer, 3 for gyroscope
    action_dim = 4  # 4 rotor speeds

    ppo_agent = PPO(state_dim, action_dim)
    num_episodes = 1000

    for episode in range(num_episodes):
        total_reward, steps = run_simulation_with_ppo(ppo_agent)
        ppo_agent.update()

        print(f"Episode {episode + 1}, Total Reward: {total_reward}, Steps: {steps}")

    return ppo_agent

if __name__ == "__main__":
    trained_agent = train_ppo()
    print("Training complete.")