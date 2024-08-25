import numpy as np

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

def create_state():
    return {
        'omega': np.array([omega_stable] * 4, dtype=np.float64),
        'angular_velocity_B': np.zeros(3, dtype=np.float64),
        'linear_velocity_W': np.zeros(3, dtype=np.float64),
        'linear_position_W': np.zeros(3, dtype=np.float64),  # Changed to float64
        'R_W_B': np.eye(3, dtype=np.float64)
    }

def create_random_action():
    return np.random.uniform(omega_min, omega_max, 4)

def step(state, action):
    omega = np.clip(action, omega_min, omega_max)
    angular_velocity_B = state['angular_velocity_B']
    linear_velocity_W = state['linear_velocity_W']
    linear_position_W = state['linear_position_W']
    R_W_B = state['R_W_B']

    F = k_f * omega * np.abs(omega)
    M = k_m * omega * np.abs(omega)

    f_B_thrust = np.array([0, np.sum(F), 0])

    tau_B_drag = np.array([0, M[0] - M[1] + M[2] - M[3], 0])
    tau_B_thrust = np.array([
        L * (F[0] - F[1] - F[2] + F[3]),
        0,
        L * (-F[0] - F[1] + F[2] + F[3])
    ])
    tau_B = tau_B_drag + tau_B_thrust

    linear_acceleration_W = np.array([0, -g * m, 0]) + R_W_B @ f_B_thrust
    linear_acceleration_W /= m

    angular_acceleration_B = np.linalg.inv(loc_I_mat) @ (
        -np.cross(angular_velocity_B, loc_I_mat @ angular_velocity_B) + tau_B
    )

    linear_velocity_W += dt * linear_acceleration_W
    linear_position_W += dt * linear_velocity_W
    angular_velocity_B += dt * angular_acceleration_B

    R_W_B += dt * R_W_B @ np.array([
        [0, -angular_velocity_B[2], angular_velocity_B[1]],
        [angular_velocity_B[2], 0, -angular_velocity_B[0]],
        [-angular_velocity_B[1], angular_velocity_B[0], 0]
    ])

    new_state = {
        'omega': omega,
        'angular_velocity_B': angular_velocity_B,
        'linear_velocity_W': linear_velocity_W,
        'linear_position_W': linear_position_W,
        'R_W_B': R_W_B
    }

    return new_state, 0

def print_state(state):
    print("  Omega:", state['omega'])
    print("  Angular Velocity:", state['angular_velocity_B'])
    print("  Linear Velocity:", state['linear_velocity_W'])
    print("  Position:", state['linear_position_W'])
    print("  Rotation Matrix:")
    print(state['R_W_B'])
