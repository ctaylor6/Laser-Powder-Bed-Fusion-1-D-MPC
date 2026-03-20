import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.signal import cont2discrete

# ─── System parameters ────────────────────────────────────────────────────────
m, k, c = 1.0, 10.0, 1.0         # mass (kg), spring (N/m), damping (N·s/m)
dt, T = 0.05, 10.0               # time step (s), total sim time (s)
N_sim = int(T / dt)

# External disturbance
def external_force(t):
    # Base sinusoid
    f = 2.0 * np.sin(2*np.pi*0.5 * t)
    # Add a +5 N step at halfway point
    # if t >= T/2:
    #     f += 10.0
    return f


# Desired trajectory
def desired_position(t):
    base = 0.5 * np.sin(2 * np.pi * 0.2 * t)
    if t >= T/2:
        return base + 0.1
    else:
        return base

# ─── Discretize continuous‐time plant ────────────────────────────────────────
A_c = np.array([[0, 1], [-k/m, -c/m]])
B_c = np.array([[0], [1/m]])
C_c = np.eye(2)
D_c = np.zeros((2,1))
Ad, Bd, _, _, _ = cont2discrete((A_c, B_c, C_c, D_c), dt)

# ─── MPC setup ───────────────────────────────────────────────────────────────
Np = 20
Q  = np.diag([100, 1])
R  = 0.1   # scalar weight

# ─── PID setup ──────────────────────────────────────────────────────────────
Kp, Ki, Kd = 150.0, 10.0, 5.0
integral, prev_error = 0.0, 0.0

# ─── Storage ────────────────────────────────────────────────────────────────
x_mpc = np.zeros((2,1))
x_pid = np.zeros((2,1))
x_mpc_log, x_pid_log = [], []
u_mpc_log, u_pid_log = [], []
ref_log = []

# ─── Simulation loop ────────────────────────────────────────────────────────
for i in range(N_sim):
    t = i*dt
    F_ext = external_force(t)
    ref = desired_position(t)

    # — PID control law —
    err = ref - x_pid[0,0]
    integral += err * dt
    deriv    = (err - prev_error) / dt
    u_pid    = Kp*err + Ki*integral + Kd*deriv
    prev_error = err

    # apply PID to plant
    x_pid = Ad @ x_pid + Bd * (u_pid + F_ext)

    # — MPC control law —
    x0 = x_mpc.copy()
    x_var = cp.Variable((2, Np+1))
    u_var = cp.Variable((1, Np))

    cost   = 0
    constr = [x_var[:,0] == x0.flatten()]

    for k_i in range(Np):
        ref_k = desired_position(t + k_i*dt)
        cost += cp.quad_form(x_var[:,k_i] - np.array([ref_k, 0]), Q)
        cost += R * cp.square(u_var[:,k_i])
        constr += [
            x_var[:,k_i+1]
            == Ad @ x_var[:,k_i]
               + Bd.flatten() * (u_var[:,k_i] + F_ext)
        ]

    prob = cp.Problem(cp.Minimize(cost), constr)
    prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)

    # take first MPC action
    u_mpc = u_var[:,0].value
    if u_mpc is None:
        u_mpc = np.array([0.0])
    x_mpc = Ad @ x_mpc + Bd * (u_mpc + F_ext)

    # ── log everything ───────────────────────────────────────────────
    x_pid_log.append(x_pid.flatten())
    x_mpc_log.append(x_mpc.flatten())
    u_pid_log.append(u_pid)
    u_mpc_log.append(u_mpc)
    ref_log.append(ref)

# ─── Convert to arrays ───────────────────────────────────────────────────────
x_pid_log = np.array(x_pid_log)
x_mpc_log = np.array(x_mpc_log)
u_pid_log = np.array(u_pid_log).flatten()
u_mpc_log = np.array(u_mpc_log).flatten()
t_vec      = np.arange(N_sim)*dt
ref_log    = np.array(ref_log)

# ─── Plot results ───────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
fig, axs = plt.subplots(2,1, figsize=(10,6), sharex=True)

# Position
axs[0].plot(t_vec, x_pid_log[:,0],   label="PID Position",   linewidth=2)
axs[0].plot(t_vec, x_mpc_log[:,0],   label="MPC Position",   linewidth=2)
axs[0].plot(t_vec, ref_log, '--',    label="Reference",       linewidth=2)
axs[0].set_ylabel("Position (m)")
axs[0].legend()
axs[0].set_title("Tracking Comparison: PID vs. MPC")

# Control effort
axs[1].plot(t_vec, u_pid_log,        label="PID Effort",      linewidth=2)
axs[1].plot(t_vec, u_mpc_log,        label="MPC Effort",      linewidth=2)
axs[1].set_ylabel("Control Input (N)")
axs[1].set_xlabel("Time (s)")
axs[1].legend()

plt.tight_layout()
plt.show()
