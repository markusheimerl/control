#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Function prototypes
void crossVec3f(float v1[3], float v2[3], float result[3]);
void multScalVec3f(float s, float v[3], float result[3]);
void addVec3f(float v1[3], float v2[3], float result[3]);
void multMat3f(float a[9], float b[9], float result[9]);
void multMatVec3f(float m[9], float v[3], float result[3]);
void vecToDiagMat3f(float v[3], float result[9]);
void invMat3f(float m[9], float result[9]);
void xRotMat3f(float rads, float result[9]);
void yRotMat3f(float rads, float result[9]);
void zRotMat3f(float rads, float result[9]);
void so3hat(float v[3], float result[9]);
void addMat3f(float a[9], float b[9], float result[9]);
void multScalMat3f(float s, float m[9], float result[9]);

// Constants
const float k_f = 0.0004905f;
const float k_m = 0.00004905f;
const float L = 0.25f;
const float l = (L / sqrtf(2));
const float I[3] = {0.01f, 0.02f, 0.01f};
float loc_I_mat[9];
float loc_I_mat_inv[9];
const float g = 9.81f;
const float m = 0.5f;
const float dt = 0.01f;
const float omega_min = 30.0f;
const float omega_max = 70.0f;
const float omega_stable = 50.0f;

// Global variables
float omega_1, omega_2, omega_3, omega_4;
float angular_velocity_B[3];
float linear_velocity_W[3];
float linear_position_W[3];
float R_W_B[9];

void initializeSimulation() {
    omega_1 = omega_2 = omega_3 = omega_4 = omega_stable;
    
    for (int i = 0; i < 3; i++) {
        angular_velocity_B[i] = 0.0f;
        linear_velocity_W[i] = 0.0f;
    }
    
    linear_position_W[0] = 0.0f;
    linear_position_W[1] = 1.0f;
    linear_position_W[2] = 0.0f;

    float xRot[9], yRot[9], zRot[9], temp[9];
    xRotMat3f(0, xRot);
    yRotMat3f(0, yRot);
    zRotMat3f(0, zRot);
    multMat3f(xRot, yRot, temp);
    multMat3f(temp, zRot, R_W_B);

    vecToDiagMat3f(I, loc_I_mat);
    invMat3f(loc_I_mat, loc_I_mat_inv);
}

void runSimulation() {
    // Limit motor speeds
    omega_1 = fmaxf(fminf(omega_1, omega_max), omega_min);
    omega_2 = fmaxf(fminf(omega_2, omega_max), omega_min);
    omega_3 = fmaxf(fminf(omega_3, omega_max), omega_min);
    omega_4 = fmaxf(fminf(omega_4, omega_max), omega_min);

    // Forces and moments
    float F1 = k_f * omega_1 * fabsf(omega_1);
    float F2 = k_f * omega_2 * fabsf(omega_2);
    float F3 = k_f * omega_3 * fabsf(omega_3);
    float F4 = k_f * omega_4 * fabsf(omega_4);

    float M1 = k_m * omega_1 * fabsf(omega_1);
    float M2 = k_m * omega_2 * fabsf(omega_2);
    float M3 = k_m * omega_3 * fabsf(omega_3);
    float M4 = k_m * omega_4 * fabsf(omega_4);

    // Thrust
    float f_B_thrust[3] = {0, F1 + F2 + F3 + F4, 0};

    // Torque
    float tau_B_drag[3] = {0, M1 - M2 + M3 - M4, 0};
    float tau_B_thrust_1[3], tau_B_thrust_2[3], tau_B_thrust_3[3], tau_B_thrust_4[3];
    float temp1[3] = {-L, 0, L}, temp2[3] = {0, F1, 0};
    crossVec3f(temp1, temp2, tau_B_thrust_1);
    
    temp1[0] = L; temp2[1] = F2;
    crossVec3f(temp1, temp2, tau_B_thrust_2);
    
    temp1[2] = -L; temp2[1] = F3;
    crossVec3f(temp1, temp2, tau_B_thrust_3);
    
    temp1[0] = -L; temp2[1] = F4;
    crossVec3f(temp1, temp2, tau_B_thrust_4);

    float tau_B_thrust[3], tau_B[3];
    addVec3f(tau_B_thrust_1, tau_B_thrust_2, tau_B_thrust);
    addVec3f(tau_B_thrust, tau_B_thrust_3, tau_B_thrust);
    addVec3f(tau_B_thrust, tau_B_thrust_4, tau_B_thrust);
    addVec3f(tau_B_drag, tau_B_thrust, tau_B);

    // Accelerations
    float linear_acceleration_W[3], temp3[3];
    multMatVec3f(R_W_B, f_B_thrust, temp3);
    addVec3f((float[3]){0, -g * m, 0}, temp3, linear_acceleration_W);
    multScalVec3f(1 / m, linear_acceleration_W, linear_acceleration_W);

    float angular_acceleration_B[3], temp4[3], temp5[3];
    multMatVec3f(loc_I_mat, angular_velocity_B, temp4);
    multScalVec3f(-1, angular_velocity_B, temp5);
    crossVec3f(temp5, temp4, temp3);
    addVec3f(temp3, tau_B, angular_acceleration_B);
    angular_acceleration_B[0] /= I[0];
    angular_acceleration_B[1] /= I[1];
    angular_acceleration_B[2] /= I[2];

    // Advance state
    float temp6[3];
    multScalVec3f(dt, linear_acceleration_W, temp6);
    addVec3f(linear_velocity_W, temp6, linear_velocity_W);
    
    multScalVec3f(dt, linear_velocity_W, temp6);
    addVec3f(linear_position_W, temp6, linear_position_W);
    
    multScalVec3f(dt, angular_acceleration_B, temp6);
    addVec3f(angular_velocity_B, temp6, angular_velocity_B);

    float temp7[9], temp8[9], temp9[9];
    so3hat(angular_velocity_B, temp7);
    multMat3f(R_W_B, temp7, temp8);
    multScalMat3f(dt, temp8, temp9);
    addMat3f(R_W_B, temp9, R_W_B);
}

int main() {
    initializeSimulation();

    for (int iteration = 0; iteration < 1000; iteration++) {
        runSimulation();
        
        printf("Iteration %d:\n", iteration);
        printf("Position: [%.2f, %.2f, %.2f]\n", linear_position_W[0], linear_position_W[1], linear_position_W[2]);
        printf("Velocity: [%.2f, %.2f, %.2f]\n", linear_velocity_W[0], linear_velocity_W[1], linear_velocity_W[2]);
        printf("Angular Velocity: [%.2f, %.2f, %.2f]\n", angular_velocity_B[0], angular_velocity_B[1], angular_velocity_B[2]);
        printf("---\n");
    }

    printf("Simulation complete.\n");
    return 0;
}

void crossVec3f(float v1[3], float v2[3], float result[3]) {
    result[0] = v1[1] * v2[2] - v1[2] * v2[1];
    result[1] = v1[2] * v2[0] - v1[0] * v2[2];
    result[2] = v1[0] * v2[1] - v1[1] * v2[0];
}

void multScalVec3f(float s, float v[3], float result[3]) {
    result[0] = v[0] * s;
    result[1] = v[1] * s;
    result[2] = v[2] * s;
}

void addVec3f(float v1[3], float v2[3], float result[3]) {
    result[0] = v1[0] + v2[0];
    result[1] = v1[1] + v2[1];
    result[2] = v1[2] + v2[2];
}

void multMat3f(float a[9], float b[9], float result[9]) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            result[i*3 + j] = a[i*3] * b[j] + a[i*3 + 1] * b[3 + j] + a[i*3 + 2] * b[6 + j];
        }
    }
}

void multMatVec3f(float m[9], float v[3], float result[3]) {
    result[0] = m[0] * v[0] + m[1] * v[1] + m[2] * v[2];
    result[1] = m[3] * v[0] + m[4] * v[1] + m[5] * v[2];
    result[2] = m[6] * v[0] + m[7] * v[1] + m[8] * v[2];
}

void vecToDiagMat3f(float v[3], float result[9]) {
    for (int i = 0; i < 9; i++) {
        result[i] = 0.0f;
    }
    result[0] = v[0];
    result[4] = v[1];
    result[8] = v[2];
}

void invMat3f(float m[9], float result[9]) {
    float det = m[0] * (m[4] * m[8] - m[7] * m[5]) -
                m[1] * (m[3] * m[8] - m[5] * m[6]) +
                m[2] * (m[3] * m[7] - m[4] * m[6]);

    if (det == 0) {
        // Handle non-invertible matrix
        for (int i = 0; i < 9; i++) {
            result[i] = 0.0f;
        }
        return;
    }

    float invDet = 1.0f / det;

    result[0] = invDet * (m[4] * m[8] - m[7] * m[5]);
    result[1] = invDet * (m[2] * m[7] - m[1] * m[8]);
    result[2] = invDet * (m[1] * m[5] - m[2] * m[4]);
    result[3] = invDet * (m[5] * m[6] - m[3] * m[8]);
    result[4] = invDet * (m[0] * m[8] - m[2] * m[6]);
    result[5] = invDet * (m[3] * m[2] - m[0] * m[5]);
    result[6] = invDet * (m[3] * m[7] - m[6] * m[4]);
    result[7] = invDet * (m[6] * m[1] - m[0] * m[7]);
    result[8] = invDet * (m[0] * m[4] - m[3] * m[1]);
}

void xRotMat3f(float rads, float result[9]) {
    float s = sinf(rads);
    float c = cosf(rads);
    result[0] = 1.0f; result[1] = 0.0f; result[2] = 0.0f;
    result[3] = 0.0f; result[4] = c;    result[5] = -s;
    result[6] = 0.0f; result[7] = s;    result[8] = c;
}

void yRotMat3f(float rads, float result[9]) {
    float s = sinf(rads);
    float c = cosf(rads);
    result[0] = c;    result[1] = 0.0f; result[2] = s;
    result[3] = 0.0f; result[4] = 1.0f; result[5] = 0.0f;
    result[6] = -s;   result[7] = 0.0f; result[8] = c;
}

void zRotMat3f(float rads, float result[9]) {
    float s = sinf(rads);
    float c = cosf(rads);
    result[0] = c;    result[1] = -s;   result[2] = 0.0f;
    result[3] = s;    result[4] = c;    result[5] = 0.0f;
    result[6] = 0.0f; result[7] = 0.0f; result[8] = 1.0f;
}

void so3hat(float v[3], float result[9]) {
    result[0] = 0.0f;  result[1] = -v[2]; result[2] = v[1];
    result[3] = v[2];  result[4] = 0.0f;  result[5] = -v[0];
    result[6] = -v[1]; result[7] = v[0];  result[8] = 0.0f;
}

void addMat3f(float a[9], float b[9], float result[9]) {
    for (int i = 0; i < 9; i++) {
        result[i] = a[i] + b[i];
    }
}

void multScalMat3f(float s, float m[9], float result[9]) {
    for (int i = 0; i < 9; i++) {
        result[i] = s * m[i];
    }
}