import math
import time
import random

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

# Initial state
omega_1 = omega_stable
omega_2 = omega_stable
omega_3 = omega_stable
omega_4 = omega_stable

angular_velocity_B = [0, 0, 0]
linear_velocity_W = [0, 0, 0]
linear_position_W = [0, 1, 0]

R_W_B = multMat3f(multMat3f(xRotMat3f(0), yRotMat3f(0)), zRotMat3f(0))

iteration = 0
max_iterations = 1000  # Run for 1000 iterations

def add_noise(value, std_dev):
    return value + random.gauss(0, std_dev)

def run_simulation():
    global omega_1, omega_2, omega_3, omega_4, angular_velocity_B, linear_velocity_W, linear_position_W, R_W_B, iteration

    # Limit motor speeds
    omega_1 = max(min(omega_1, omega_max), omega_min)
    omega_2 = max(min(omega_2, omega_max), omega_min)
    omega_3 = max(min(omega_3, omega_max), omega_min)
    omega_4 = max(min(omega_4, omega_max), omega_min)

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
    angular_acceleration_B = multMatVec3f(loc_I_mat_inv, 
                                          addVec3f(tau_B, 
                                                   crossVec3f(multScalVec3f(-1, angular_velocity_B), 
                                                              multMatVec3f(loc_I_mat, angular_velocity_B))))

    # Calculate linear acceleration in body frame (excluding gravity)
    linear_acceleration_B = multMatVec3f(invMat3f(R_W_B), linear_acceleration_W)

    # Simulate IMU readings with noise
    imu_accel_reading = [add_noise(acc, accel_noise_std) for acc in linear_acceleration_B]
    imu_gyro_reading = [add_noise(ang_vel, gyro_noise_std) for ang_vel in angular_velocity_B]

    # Advance state
    linear_velocity_W = addVec3f(linear_velocity_W, multScalVec3f(dt, linear_acceleration_W))
    linear_position_W = addVec3f(linear_position_W, multScalVec3f(dt, linear_velocity_W))
    angular_velocity_B = addVec3f(angular_velocity_B, multScalVec3f(dt, angular_acceleration_B))
    R_W_B = addMat3f(R_W_B, multScalMat3f(dt, multMat3f(R_W_B, so3hat(angular_velocity_B))))

    print(f"Iteration {iteration}:")
    print(f"Position: {linear_position_W}")
    print(f"Velocity: {linear_velocity_W}")
    print(f"Angular Velocity: {angular_velocity_B}")
    print(f"IMU Accelerometer Reading: {imu_accel_reading}")
    print(f"IMU Gyroscope Reading: {imu_gyro_reading}")
    print('---')

    iteration += 1
    if iteration >= max_iterations:
        print("Simulation complete.")
        return False
    return True

# Run the simulation
while run_simulation():
    pass