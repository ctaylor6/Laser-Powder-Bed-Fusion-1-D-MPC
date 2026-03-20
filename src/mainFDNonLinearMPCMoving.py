import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from solve_pde_fd import solve_pde_fd  
import os
import platform
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use('Agg')  # For headless environments
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import rc

def clear_terminal():
    if platform.system() == 'Windows':
        os.system('cls')
    else:
        os.system('clear')
clear_terminal()  # Call it at the start

def rhs_pde(x, t, P_scalar, t0=0.0):
    r = 0.01   # Beam width
    v = 20  # Scan speed (units of x per unit time)

    # Shift the local time array by t0 so the Gaussian moves
    X, T = np.meshgrid(x, t + t0, indexing='ij')  # X: (len(x), len(t)), T likewise

    Q = P_scalar * np.exp(-2 * (X - v * T)**2 / r**2)
    return Q

# ─── Plant step function (returns max temperature at step k) ────────────────
def pde_step(u_in, temp_profile, t0):
    """
    u_in: control power at time t0
    temp_profile: 1D array of temperature at time t0
    t0: global time at start of this step
    """
    # build the 2-level source Q over [t0, t0+dt]
    Q_step = rhs_pde(x, t_step, u_in, t0=t0)

    _, _, U_step = solve_pde_fd(
        Q_step,
        x, t_step,
        initial_condition=temp_profile
    )
    new_profile = U_step[-1, :]
    return new_profile, np.max(new_profile)

# ─── MPC function (returns max temperature at step k) ────────────────
def simulate_plant_over_horizon(u_seq, init_profile, t0):
    """
    u_seq      : list or array of length Np
    init_profile: temperature profile at current time
    t0         : current absolute time (i.e. t_k)
    Returns a list of max‐temps at [t₀, t₀+Δt, …, t₀+Np·Δt] (length Np+1)
    """
    x_pred    = [np.max(init_profile)]
    temp_prof = init_profile.copy()
    t_curr    = t0.copy()  # start at t0

    for u_j in u_seq:
        # step the PDE using the correct time offset
        temp_prof, xj = pde_step(u_j, temp_prof, t0=t_curr)
        x_pred.append(xj)

        # advance time by one step
        t_curr += dt

    return x_pred

# ─── Grid and initial solve (for shape info) ─────────────────────────────────
x = np.linspace(0, 1, 1000)
t = np.linspace(0, 0.025, 250)
# run once to get U_fdm shape; we’ll ignore U_fdm here
x, t, U_tmp = solve_pde_fd(rhs_pde(x, t.reshape(-1),1), x, t,
                        initial_condition=lambda x: 0,
                        )
num_time_steps = U_tmp.shape[0]

# Define grid for PDE
temp_profile = np.array([0.0] * len(x))  # initial condition

# ─── PID parameters ─────────────────────────
# PID parameters
Kp, Ki, Kd = 10000.0, 100000000.0, 0.0000000001

# initialize PID state
temp_profile_pid = temp_profile.copy()
error_int = 0.0
prev_error = 0.0

# ─── MPC parameters ──────────────────────────────────────────────────────────
dt     = t[1] - t[0]           # your existing Δt
t_step = np.array([0.0, dt])  # just two time‐levels: t=0 and t=Δt

Np = 5           # prediction horizon (in steps)
Q = 10000000          # state tracking weight
R = 0.0001         # control effort weight
alpha = 800 * Q   # terminal‐state weight
S     = 200 * R  # rate‐penalty weight (tune as needed)

# desired max‐temperature trajectory
# def desired_temp(time):
#     if time >= 0.5 * np.max(t): #and time <= 0.75 * np.max(t):
#          return 0.05
#     else:
#         return 0.025
def desired_temp(time):
    """
    Ramps linearly from base to peak and back over [0, t_max],
    with a brief disturbance (spike) at t = 0.5 * t_max.
    """
    t_max = np.max(t)
    t_mid = 0.5 * t_max
    base = 0.025
    peak = 0.075
    disturbance = 0.015  # spike magnitude

    # Linear ramp up to the midpoint, then ramp down
    if time <= t_mid:
        ref = base + (peak - base) * (time / t_mid)
    else:
        ref = peak - (peak - base) * ((time - t_mid) / t_mid)

    # Add a short disturbance exactly at t_mid (within one time step)
    if abs(time - t_mid) < (t[1] - t[0]):
        ref += disturbance

    return ref

# ─── Simulation storage ─────────────────────────────────────────────────────
x_temp = np.array([[np.max(temp_profile)]])  # Use actual initial max temp

# Initialize logs with the true initial condition (so plots start at t=0)
t_log     = [0.0]
x_log     = [np.max(temp_profile)]
u_log     = [0.0]
x_pid_log = [np.max(temp_profile_pid)]
u_pid_log = [0.0]

# ─── Main simulation loop ──────────────────────────────────────────────────
for k in range(num_time_steps):
    
    # Prepare to store temperature profiles for GIF 
    if k == 0:
        U_frames = []
        
    # Timing variables for the simulation
    t_k = k * dt
    ref = desired_temp(t_k)

    # set up MPC decision variables
    x_var = cp.Variable((1, Np+1))
    u_var = cp.Variable((1, Np))
    cost = 0
    constr = [x_var[:, 0] == x_temp.flatten()]

    # ─── MPC decision ────────────────────────────
    # get last applied control (for Δu penalty)
    u_prev = u_log[-1] if u_log else 0.0

    # Precompute the reference trajectory for Np+1 points
    ref_seq_full = [desired_temp(t_k + j*dt) for j in range(Np+1)]

    # now a single-argument cost that closes over u_prev and ref_seq_full
    def cost_for_u0(u0):
        x_pred = simulate_plant_over_horizon([u0]*Np, temp_profile,t0=t_k)

        # 1) tracking cost
        J = sum(Q*(x_pred[j] - ref_seq_full[j])**2 for j in range(1, Np+1))
        # 2) terminal‐state penalty
        J += alpha * (x_pred[-1] - ref_seq_full[-1])**2
        # 3) control‐effort penalty (first move only)
        J += R * (u0**2)
        # 4) control‐rate penalty
        J += S * (u0 - u_prev)**2
        return J

    # minimize over u0 ∈ [0, 5000]
    res = minimize_scalar(cost_for_u0, bounds=(0, 5000), method='bounded')
    best_u, min_cost = res.x, res.fun

    print(f"\nChosen u = {best_u:.3f} (cost={min_cost:.4f})")

    # ─── Apply the best control for one Δt step ────────────────────────────
    # Apply the MPC step
    temp_profile, x_temp_val = pde_step(best_u, temp_profile,t0=t_k)
    x_temp = np.array([[x_temp_val]])
    u_log.append(best_u)
    x_log.append(x_temp_val)
    t_log.append(t_k + dt)   # log the time *after* the step
    
    # ─── PID benchmark step ─────────────────────────────────────────────────
    # constants
    SAT_LOW, SAT_HIGH = 0.0, 5000.0
    epsilon = 1e-6

    current_pid_temp = np.max(temp_profile_pid)
    error = ref - current_pid_temp # compute error
    error_int += error * dt # integrator
    error_der = (error - prev_error) / dt # derivative
    u_pid = Kp*error + Ki*error_int + Kd*error_der # raw PID

    #apply saturation
    u_pid_clipped = np.clip(u_pid, SAT_LOW, SAT_HIGH)

    #anti-windup: if clipped, undo integrator increment
    if u_pid_clipped <= SAT_LOW + epsilon or u_pid_clipped >= SAT_HIGH - epsilon:
        error_int -= error * dt

    u_pid = u_pid_clipped

    # plant step
    temp_profile_pid, x_pid_val = pde_step(u_pid, temp_profile_pid,t0=t_k)
    prev_error = error

    # logging
    u_pid_log.append(u_pid)
    x_pid_log.append(x_pid_val)
    
    # Save frame every 10 steps to reduce data size
    if k % 1 == 0:
        U_frames.append(temp_profile.copy())
    
# ─── After the loop, convert logs to numpy arrays ────────────────────────────
t_log     = np.array(t_log)
x_log     = np.array(x_log)
u_log     = np.array(u_log)
x_pid_log = np.array(x_pid_log)
u_pid_log = np.array(u_pid_log)

# New: save directory for plots
save_dir = "Plots"
os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

# ─── Plotting ──────────────────────────────────────────────────────────────────────────
# Plot the maximum temperature
plt.figure(figsize=(5,5))
plt.plot(t_log,    x_log,     label='Max Temp (MPC)', linewidth=2)
plt.plot(t_log,    x_pid_log, label='Max Temp (PID)', linewidth=2)
plt.plot(t_log, [desired_temp(tt) for tt in t_log], '--', label='Reference', linewidth=2)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Max Temperature', fontsize=14)
plt.title('MPC vs PID Tracking of Max Temp.', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.xlim(0.01, 0.025)
plt.savefig(os.path.join(save_dir, "temperature_tracking.svg"))  # ← Save here

# Plot the control signals:
plt.figure(figsize=(10,5))
plt.plot(t_log, u_log,     label='u (MPC)', linewidth=2)
plt.plot(t_log, u_pid_log, label='u (PID)', linewidth=2)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Control Input', fontsize=14)
plt.title('Control Trajectories: MPC vs PID', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "control_signal_tracking.png"))  # ← Save here

# ─── Create Temperature Profile GIF ───
# Now animate the collected frames
fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
line_analytical, = ax.plot([], [], label='Finite Difference Solution', color='green', linewidth=2)
ax.set_title('Finite Difference', fontsize=14.5)
ax.set_xlabel('x', fontsize=16)
ax.set_ylabel(r'$u(x,t)$', fontsize=16)
ax.legend(fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=16)
# Set plot limits
ax.set_xlim(np.min(x), np.max(x))
ax.set_ylim(np.min(U_frames), np.max(U_frames))
# Optional: Time text
text = ax.text(0.02, 0.90, '', transform=ax.transAxes, fontsize=14)
# Update function
def update(frame):
    line_analytical.set_data(x, U_frames[frame])
    text.set_text(f"Frame: {frame*10}")
    return line_analytical, text
# Create animation from U_frames
anim = FuncAnimation(fig, update, frames=len(U_frames), blit=True)
# Save as GIF
anim.save(os.path.join(save_dir, 'Finite_DifferenceCDT.gif'), dpi=200, writer='pillow', fps=30)

# ─── Create Temperature Profile GIF ───-───────────────────────────────
# Compute max temperature over time
U_frames_array = np.array(U_frames)  # Convert list of arrays to a NumPy array
max_temp_over_time = np.max(U_frames_array, axis=1)  # Shape: (num_time_steps,)
time = t[:len(max_temp_over_time)]  # Ensure time matches the length of max_temp_over_time
# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 6), dpi=50)
line, = ax.plot([], [], color='red', linewidth=2, label='Melt Pool Temperature')
ax.set_title('Melt Pool Temperature Over Time', fontsize=14.5)
ax.set_xlabel('Time', fontsize=16)
ax.set_ylabel('Temperature', fontsize=16)
ax.legend(fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_xlim(np.min(time), np.max(time))
ax.set_ylim(np.min(max_temp_over_time), np.max(max_temp_over_time) * 1.05)
# Create a text label for the current max
text = ax.text(0.02, 0.85, '', transform=ax.transAxes, fontsize=16)
# Animation update function
def update(frame):
    current_time = time[:frame+1]
    current_max = max_temp_over_time[:frame+1]
    line.set_data(current_time, current_max)
    text.set_text(f'Time = {time[frame]:.2f}s\nMax T = {max_temp_over_time[frame]:.2f}')
    return line, text
# Create animation
anim = FuncAnimation(fig, update, frames=len(time), blit=True)
fig.tight_layout()  # Ensures padding is correct
# Save the animation
anim.save(os.path.join(save_dir, 'Melt_Pool_Over_TimeCDT.gif'), dpi=100, writer='pillow', fps=30)



