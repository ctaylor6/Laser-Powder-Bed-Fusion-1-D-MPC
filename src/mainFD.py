import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

from solve_pde_fd import solve_pde_fd  


# ─── PDE heat‐source definition ───────────────────────────────────────────────
def laser_power(t):
    # Hard-coded power profile — can be piecewise, sinusoidal, step, etc.
    return 100 + 10 * np.sin(3 * 2 * np.pi * t)

def rhs_pde(x, t, P_scalar):
    r = 0.01 # Beam width
    v = 0.5 # Scan speed origionally at 20
    t = t[None, :]
    x = x[:, None]

    # Use scalar control input as constant power
    P = P_scalar  # scalar value
    Q = P * np.exp(-2 * (x - v * t)**2 / r**2)
    return Q


# ─── Grid and initial solve (for shape info) ─────────────────────────────────
x = np.linspace(0, 1, 100)
t = np.linspace(0, 1, 100)
# run once to get U_fdm shape; we’ll ignore U_fdm here
_, _, U_tmp = solve_pde_fd(rhs_pde(x, t, 10), x, t, initial_condition=lambda xi: 0.01)
num_time_steps = U_tmp.shape[0]


# ─── Plant step function (returns max temperature at step k) ────────────────
def pde_step(u_in, k):
    """
    u_in : scalar control
    k    : time index (0…num_time_steps–1)
    """
    # re-solve PDE over full horizon with updated initial condition
    # (in practice you’d use a reduced-order or surrogate model!)
    Q = rhs_pde(x, t, u_in)
    _, _, U = solve_pde_fd(
        Q,
        x, t,
        initial_condition=lambda xi: 0.0
    )
    return np.max(U[k, :])

# ─── MPC parameters ──────────────────────────────────────────────────────────
dt = t[1] - t[0]
T_total = num_time_steps * dt

Np = 3           # prediction horizon (in steps)
Q = 20.0          # state tracking weight
R = 0.5           # control effort weight
alpha = 0.1      # surrogate gain: ΔT ≈ α·u

# desired max‐temperature trajectory
def desired_temp(time):
    return 0.0002  # constant target


# ─── Simulation storage ─────────────────────────────────────────────────────
x_temp = np.array([[0.01]])   # initial max‐temperature
x_log   = []
u_log   = []
t_log   = []

# ─── Main simulation loop ──────────────────────────────────────────────────
for k in range(num_time_steps):
    t_k = k * dt
    ref = desired_temp(t_k)

    # set up MPC decision variables
    x_var = cp.Variable((1, Np+1))
    u_var = cp.Variable((1, Np))

    cost = 0
    constr = [x_var[:, 0] == x_temp.flatten()]

    for j in range(Np):
        # stage cost
        cost += Q * cp.square(x_var[:, j] - ref)
        cost += R * cp.square(u_var[:, j])
        # simple linear prediction model
        constr += [x_var[:, j+1] == x_var[:, j] + alpha * u_var[:, j]]

    prob = cp.Problem(cp.Minimize(cost), constr)
    prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)

    # extract and apply control
    u_k = u_var[:, 0].value
    if u_k is None:
        u_k = np.array([0.0])
    u_k = float(u_k)

    T_next = pde_step(u_k, k)
    x_temp = np.array([[T_next]])

    #log
    x_log.append(T_next)
    u_log.append(u_k)
    t_log.append(t_k)

# ─── Plotting results ────────────────────────────────────────────────────────
t_log = np.array(t_log)
x_log = np.array(x_log)
u_log = np.array(u_log)

plt.figure(figsize=(10,5))
plt.plot(t_log, x_log, label='Max Temperature (MPC)', linewidth=2)
plt.plot(t_log, [desired_temp(tt) for tt in t_log], '--', label='Reference', linewidth=2)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Max Temperature', fontsize=14)
plt.title('MPC Tracking of Maximum Temperature', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

# ─── Open-Loop Behavior (No MPC) ─────────────────────────────────────────────
# Run same PDE with constant or zero control input to observe open-loop dynamics

u_open_loop = 0.0  # control input (change to test different values)
x_ol_log = []
t_ol_log = []

# Reset initial condition
x_temp_ol = np.array([[0.01]])

for k in range(num_time_steps):
    t_k = k * dt
    T_next_ol = pde_step(u_open_loop, k)
    x_temp_ol = np.array([[T_next_ol]])

    x_ol_log.append(T_next_ol)
    t_ol_log.append(t_k)

# Plot open-loop behavior alongside MPC result
plt.figure(figsize=(10,5))
plt.plot(t_ol_log, x_ol_log, label='Max Temp (Open Loop)', linewidth=2, linestyle='-')
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Max Temperature', fontsize=14)
plt.title('Open-Loop vs MPC Temperature Tracking', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot heating source term over time 
Q = rhs_pde(x, t, P_scalar=100)
#plt.plot(t, Q, label='Source Term', linewidth=2, linestyle='-')
plt.imshow(Q, aspect='auto', extent=[t[0], t[-1], x[0], x[-1]], origin='lower')
plt.colorbar(label='Heat Source Q(x,t)')
plt.xlabel('Time')
plt.ylabel('Position x')
plt.title('Heat Source Term Q(x,t)')
plt.show()

