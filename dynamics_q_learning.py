import math
import time
import random
import numpy as np
import tensorflow as tf
from collections import deque

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

# Constants (same as before)
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

# RL parameters
STATE_SIZE = 6  # 3 accel + 3 gyro readings
ACTION_SIZE = 4  # 4 rotor speeds
BATCH_SIZE = 32
GAMMA = 0.99
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001
MEMORY_SIZE = 10000

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.gamma = GAMMA
        self.epsilon = EPSILON
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.learning_rate = LEARNING_RATE
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_size,)),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def add_noise(value, std_dev):
    return value + random.gauss(0, std_dev)

def random_vector(max_value):
    return [random.uniform(-max_value, max_value) for _ in range(3)]

def discretize_action(action):
    # Map the discrete action to rotor speeds
    if action == 0:
        return [omega_min, omega_min, omega_min, omega_min]
    elif action == 1:
        return [omega_max, omega_min, omega_max, omega_min]
    elif action == 2:
        return [omega_min, omega_max, omega_min, omega_max]
    else:
        return [omega_max, omega_max, omega_max, omega_max]

def run_episode(agent, train=True):
    # Initialize state
    angular_velocity_B = random_vector(1.0)
    linear_velocity_W = random_vector(2.0)
    linear_position_W = [0, 1, 0]
    R_W_B = multMat3f(multMat3f(xRotMat3f(0), yRotMat3f(0)), zRotMat3f(0))

    total_reward = 0
    done = False

    for step in range(1000):  # Max 1000 steps per episode
        # Get IMU readings (current state)
        linear_acceleration_B = multMatVec3f(invMat3f(R_W_B), addVec3f([0, -g, 0], [0, 0, 0]))
        imu_accel_reading = [add_noise(acc, accel_noise_std) for acc in linear_acceleration_B]
        imu_gyro_reading = [add_noise(ang_vel, gyro_noise_std) for ang_vel in angular_velocity_B]
        
        state = np.reshape(imu_accel_reading + imu_gyro_reading, [1, STATE_SIZE])

        # Choose action
        action = agent.act(state)
        omega_1, omega_2, omega_3, omega_4 = discretize_action(action)

        # Apply action and simulate
        F1 = k_f * omega_1 * abs(omega_1)
        F2 = k_f * omega_2 * abs(omega_2)
        F3 = k_f * omega_3 * abs(omega_3)
        F4 = k_f * omega_4 * abs(omega_4)

        M1 = k_m * omega_1 * abs(omega_1)
        M2 = k_m * omega_2 * abs(omega_2)
        M3 = k_m * omega_3 * abs(omega_3)
        M4 = k_m * omega_4 * abs(omega_4)

        f_B_thrust = [0, F1 + F2 + F3 + F4, 0]

        tau_B_drag = [0, M1 - M2 + M3 - M4, 0]
        tau_B_thrust_1 = crossVec3f([-L, 0, L], [0, F1, 0])
        tau_B_thrust_2 = crossVec3f([L, 0, L], [0, F2, 0])
        tau_B_thrust_3 = crossVec3f([L, 0, -L], [0, F3, 0])
        tau_B_thrust_4 = crossVec3f([-L, 0, -L], [0, F4, 0])
        tau_B_thrust = addVec3f(tau_B_thrust_1, tau_B_thrust_2)
        tau_B_thrust = addVec3f(tau_B_thrust, tau_B_thrust_3)
        tau_B_thrust = addVec3f(tau_B_thrust, tau_B_thrust_4)
        tau_B = addVec3f(tau_B_drag, tau_B_thrust)

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
        reward = -sum([abs(v) for v in angular_velocity_B]) - sum([abs(v) for v in linear_velocity_W])
        total_reward += reward

        # Check if episode is done
        if sum([abs(v) for v in angular_velocity_B]) < 0.1 and sum([abs(v) for v in linear_velocity_W]) < 0.1:
            done = True

        # Get next state
        next_linear_acceleration_B = multMatVec3f(invMat3f(R_W_B), addVec3f([0, -g, 0], [0, 0, 0]))
        next_imu_accel_reading = [add_noise(acc, accel_noise_std) for acc in next_linear_acceleration_B]
        next_imu_gyro_reading = [add_noise(ang_vel, gyro_noise_std) for ang_vel in angular_velocity_B]
        next_state = np.reshape(next_imu_accel_reading + next_imu_gyro_reading, [1, STATE_SIZE])

        # Remember the transition
        if train:
            agent.remember(state, action, reward, next_state, done)

        if done:
            break

    return total_reward, step

# Training
agent = DQNAgent(STATE_SIZE, ACTION_SIZE)
episodes = 1000
for e in range(episodes):
    total_reward, steps = run_episode(agent)
    print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward}, Steps: {steps}")
    
    if len(agent.memory) > BATCH_SIZE:
        agent.replay(BATCH_SIZE)

# Save the trained model
agent.model.save("quadcopter_stabilization_model.h5")

# Test the trained model
test_episodes = 10
for e in range(test_episodes):
    total_reward, steps = run_episode(agent, train=False)
    print(f"Test Episode {e+1}/{test_episodes}, Total Reward: {total_reward}, Steps: {steps}")