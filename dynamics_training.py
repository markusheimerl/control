import math
import time
import random
import torch
import torch.nn as nn

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


def orthonormalize(R):
    # Gram-Schmidt Orthogonalization
    x = [R[0], R[3], R[6]]
    y = [R[1], R[4], R[7]]
    z = [R[2], R[5], R[8]]

    # Normalize x
    x_norm = math.sqrt(sum(i*i for i in x))
    x = [i/x_norm for i in x]

    # Make y orthogonal to x
    dot_xy = sum(x[i]*y[i] for i in range(3))
    y = [y[i] - dot_xy*x[i] for i in range(3)]

    # Normalize y
    y_norm = math.sqrt(sum(i*i for i in y))
    y = [i/y_norm for i in y]

    # z is the cross product of x and y
    z = crossVec3f(x, y)

    # Construct the orthonormalized matrix
    return [
        x[0], y[0], z[0],
        x[1], y[1], z[1],
        x[2], y[2], z[2]
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

def vector_norm_squared(v):
    return sum(x*x for x in v)

def calculate_reward(angular_velocity_B, linear_velocity_W):
    return -(vector_norm_squared(angular_velocity_B) + vector_norm_squared(linear_velocity_W))

# Neural Network Model
class QuadcopterController(nn.Module):
    def __init__(self):
        super(QuadcopterController, self).__init__()
        
        self.layer1 = nn.Linear(6, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 16)
        self.layer4 = nn.Linear(16, 8)
        self.layer5 = nn.Linear(8, 4)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.relu(self.layer4(x))
        x = self.layer5(x)
        return x

# Instantiate the model
model = QuadcopterController()

def get_rotor_speeds(imu_accel_reading, imu_gyro_reading):
    sensor_data = imu_accel_reading + imu_gyro_reading
    input_tensor = torch.tensor(sensor_data, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    rotor_speeds = output.squeeze().tolist()
    scaled_speeds = [omega_min + (omega_max - omega_min) * speed for speed in rotor_speeds]
    
    return scaled_speeds

def run_simulation():
    # Initialize state with random values
    angular_velocity_B = random_vector(angular_velocity_max)
    linear_velocity_W = random_vector(linear_velocity_max)
    linear_position_W = [0, 1, 0]
    R_W_B = multMat3f(multMat3f(xRotMat3f(0), yRotMat3f(0)), zRotMat3f(0))

    cumulative_reward = 0
    triplets = []

    for iteration in range(max_iterations):
        # Calculate linear acceleration in body frame (excluding gravity)
        linear_acceleration_B = multMatVec3f(invMat3f(R_W_B), [0, -g, 0])

        # Simulate IMU readings with noise
        imu_accel_reading = [add_noise(acc, accel_noise_std) for acc in linear_acceleration_B]
        imu_gyro_reading = [add_noise(ang_vel, gyro_noise_std) for ang_vel in angular_velocity_B]

        # Get rotor speeds from the model
        rotor_speeds = get_rotor_speeds(imu_accel_reading, imu_gyro_reading)
        omega_1, omega_2, omega_3, omega_4 = rotor_speeds

        # Forces and moments
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
        angular_acceleration_B = multMatVec3f(loc_I_mat_inv, addVec3f(tau_B, crossVec3f(multScalVec3f(-1, angular_velocity_B), multMatVec3f(loc_I_mat, angular_velocity_B))))

        # Advance state
        linear_velocity_W = addVec3f(linear_velocity_W, multScalVec3f(dt, linear_acceleration_W))
        linear_position_W = addVec3f(linear_position_W, multScalVec3f(dt, linear_velocity_W))
        angular_velocity_B = addVec3f(angular_velocity_B, multScalVec3f(dt, angular_acceleration_B))
        R_W_B = addMat3f(R_W_B, multScalMat3f(dt, multMat3f(R_W_B, so3hat(angular_velocity_B))))
        R_W_B = orthonormalize(R_W_B)

        # Calculate reward
        reward = calculate_reward(angular_velocity_B, linear_velocity_W)
        cumulative_reward += reward

        # Record state + action + reward triplet
        triplet = imu_accel_reading + imu_gyro_reading + rotor_speeds + [reward]
        triplets.append(triplet)

        # Print state (you might want to comment this out or modify for large number of simulations)
        print(f"Iteration {iteration}:")
        print(f"Position: {linear_position_W}")
        print(f"Velocity: {linear_velocity_W}")
        print(f"Angular Velocity: {angular_velocity_B}")
        print(f"IMU Accelerometer Reading: {imu_accel_reading}")
        print(f"IMU Gyroscope Reading: {imu_gyro_reading}")
        print(f"Rotor Speeds: {omega_1:.2f}, {omega_2:.2f}, {omega_3:.2f}, {omega_4:.2f}")
        print(f"Reward: {reward}")
        print(f"Cumulative Reward: {cumulative_reward}")
        print('---')

    return cumulative_reward, triplets

# Run multiple simulations
simulation_rewards = []
all_triplets = []

for sim in range(num_simulations):
    print(f"Starting simulation {sim + 1}")
    cumulative_reward, triplets = run_simulation()
    simulation_rewards.append(cumulative_reward)
    all_triplets.extend(triplets)
    print(f"Simulation {sim + 1} complete")
    print(f"Final Cumulative Reward: {cumulative_reward}")
    print("============================")

# Print summary statistics
average_reward = sum(simulation_rewards) / num_simulations
print(f"Average Cumulative Reward: {average_reward}")
print(f"Best Cumulative Reward: {max(simulation_rewards)}")
print(f"Worst Cumulative Reward: {min(simulation_rewards)}")