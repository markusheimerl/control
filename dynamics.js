function crossVec3f(v1, v2) {
    return [
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0]
    ];
}

function multScalVec3f(s, v) {
    return [v[0] * s, v[1] * s, v[2] * s];
}

function addVec3f(v1, v2) {
    return [v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]];
}

function multMat3f(a, b) {
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
    ];
}

function multMatVec3f(m, v) {
    return [
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
        m[6] * v[0] + m[7] * v[1] + m[8] * v[2]
    ];
}

function vecToDiagMat3f(v) {
    return [
        v[0], 0.0, 0.0,
        0.0, v[1], 0.0,
        0.0, 0.0, v[2]
    ];
}

function invMat3f(m) {
    let det =
        m[0] * (m[4] * m[8] - m[7] * m[5]) -
        m[1] * (m[3] * m[8] - m[5] * m[6]) +
        m[2] * (m[3] * m[7] - m[4] * m[6]);

    if (det === 0) {
        throw new Error("Matrix is not invertible");
    }

    let invDet = 1.0 / det;

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
    ];
}

function xRotMat3f(rads) {
    let s = Math.sin(rads);
    let c = Math.cos(rads);
    return [
        1.0, 0.0, 0.0,
        0.0, c, -s,
        0.0, s, c
    ];
}

function yRotMat3f(rads) {
    let s = Math.sin(rads);
    let c = Math.cos(rads);
    return [
        c, 0.0, s,
        0.0, 1.0, 0.0,
        -s, 0.0, c
    ];
}

function zRotMat3f(rads) {
    let s = Math.sin(rads);
    let c = Math.cos(rads);
    return [
        c, -s, 0.0,
        s, c, 0.0,
        0.0, 0.0, 1.0
    ];
}

function so3hat(v) {
    return [
        0.0, -v[2], v[1],
        v[2], 0.0, -v[0],
        -v[1], v[0], 0.0,
    ];
}

function addMat3f(a, b) {
    return [
        a[0] + b[0], a[1] + b[1], a[2] + b[2],
        a[3] + b[3], a[4] + b[4], a[5] + b[5],
        a[6] + b[6], a[7] + b[7], a[8] + b[8]
    ];
}

function multScalMat3f(s, m) {
    return [
        s * m[0], s * m[1], s * m[2],
        s * m[3], s * m[4], s * m[5],
        s * m[6], s * m[7], s * m[8]
    ];
}

// ----------------------------------- CONSTANTS -----------------------------------
const k_f = 0.0004905;
const k_m = 0.00004905;
const L = 0.25;
const l = (L / Math.sqrt(2));
const I = [0.01, 0.02, 0.01];
const loc_I_mat = vecToDiagMat3f(I);
const loc_I_mat_inv = invMat3f(loc_I_mat);
const g = 9.81;
const m = 0.5;
const dt = 0.01;
const omega_min = 30;
const omega_max = 70;
const omega_stable = 50;

// ----------------------------------- DYNAMICS -----------------------------------
let omega_1 = omega_stable;
let omega_2 = omega_stable;
let omega_3 = omega_stable;
let omega_4 = omega_stable;

let angular_velocity_B = [0, 0, 0];
let linear_velocity_W = [0, 0, 0];
let linear_position_W = [0, 1, 0];

let R_W_B = multMat3f(multMat3f(xRotMat3f(0), yRotMat3f(0)), zRotMat3f(0));

setInterval(function () {
	// --- LIMIT MOTOR SPEEDS ---
	omega_1 = Math.max(Math.min(omega_1, omega_max), omega_min);
	omega_2 = Math.max(Math.min(omega_2, omega_max), omega_min);
	omega_3 = Math.max(Math.min(omega_3, omega_max), omega_min);
	omega_4 = Math.max(Math.min(omega_4, omega_max), omega_min);

	// --- FORCES AND MOMENTS ---
	let F1 = k_f * omega_1 * Math.abs(omega_1);
	let F2 = k_f * omega_2 * Math.abs(omega_2);
	let F3 = k_f * omega_3 * Math.abs(omega_3);
	let F4 = k_f * omega_4 * Math.abs(omega_4);

	let M1 = k_m * omega_1 * Math.abs(omega_1);
	let M2 = k_m * omega_2 * Math.abs(omega_2);
	let M3 = k_m * omega_3 * Math.abs(omega_3);
	let M4 = k_m * omega_4 * Math.abs(omega_4);

	// --- THRUST ---
	let f_B_thrust = [0, F1 + F2 + F3 + F4, 0];

	// --- TORQUE ---
	let tau_B_drag = [0, M1 - M2 + M3 - M4, 0];
	let tau_B_thrust_1 = crossVec3f([-L, 0, L], [0, F1, 0]);
	let tau_B_thrust_2 = crossVec3f([L, 0, L], [0, F2, 0]);
	let tau_B_thrust_3 = crossVec3f([L, 0, -L], [0, F3, 0]);
	let tau_B_thrust_4 = crossVec3f([-L, 0, -L], [0, F4, 0]);
	let tau_B_thrust = addVec3f(tau_B_thrust_1, tau_B_thrust_2);
	tau_B_thrust = addVec3f(tau_B_thrust, tau_B_thrust_3);
	tau_B_thrust = addVec3f(tau_B_thrust, tau_B_thrust_4);
	let tau_B = addVec3f(tau_B_drag, tau_B_thrust);

	// --- ACCELERATIONS ---
	let linear_acceleration_W = addVec3f([0, -g * m, 0], multMatVec3f(R_W_B, f_B_thrust));
	linear_acceleration_W = multScalVec3f(1 / m, linear_acceleration_W);
	let angular_acceleration_B = addVec3f(crossVec3f(multScalVec3f(-1, angular_velocity_B), multMatVec3f(loc_I_mat, angular_velocity_B)), tau_B);
	angular_acceleration_B[0] = angular_acceleration_B[0] / I[0];
	angular_acceleration_B[1] = angular_acceleration_B[1] / I[1];
	angular_acceleration_B[2] = angular_acceleration_B[2] / I[2];

	// --- ADVANCE STATE ---
	linear_velocity_W = addVec3f(linear_velocity_W, multScalVec3f(dt, linear_acceleration_W));
	linear_position_W = addVec3f(linear_position_W, multScalVec3f(dt, linear_velocity_W));
	angular_velocity_B = addVec3f(angular_velocity_B, multScalVec3f(dt, angular_acceleration_B));
	R_W_B = addMat3f(R_W_B, multScalMat3f(dt, multMat3f(R_W_B, so3hat(angular_velocity_B))));

}, dt);