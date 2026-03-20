import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from solve_pde_fd import solve_pde_fd  
import os
import platform
from scipy.optimize import minimize_scalar
import matplotlib
#matplotlib.use('Agg')  # For headless environments
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import rc
import time

# This script simulates a moving laser beam on a material, tracking the melt pool temperature using Model Predictive Control (MPC) and comparing it to a PID controller.
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
    u_seq       : list or array of length Np
    init_profile: temperature profile at current time (1D array over x)
    t0          : current absolute time (i.e. t_k)
    Returns a list of *melt-pool* temperatures at 
    [t0, t0+Δt, …, t0+Np·Δt] (length Np+1)
    """
    # 1) first entry is the melt-pool temp at init
    melt_temps = [ sample_melt_pool(init_profile, x, t0, v) ]
    
    temp_prof = init_profile.copy()
    t_curr    = t0

    for u_j in u_seq:
        # 2) advance the PDE one step using the moving source
        temp_prof, _ = pde_step(u_j, temp_prof, t0=t_curr)
        
        # 3) sample the melt-pool UNDER the laser now at t_curr
        melt = sample_melt_pool(temp_prof, x, t_curr, v)
        melt_temps.append(melt)

        # 4) advance time
        t_curr += dt

    return melt_temps

# ─── Sample melt pool temperature at laser position ────────────────────────
def sample_melt_pool(temp_profile, x, t_k, v):
    """
    temp_profile: 1D array of T(x) at time t_k
    x:            1D spatial grid
    t_k:          current time
    v:            laser scan speed
    """
    x_laser = v * t_k
    # clamp to domain
    x_laser = np.clip(x_laser, x.min(), x.max())
    # linear interpolation
    return np.interp(x_laser, x, temp_profile)

# ─── Grid and initial solve (for shape info) ──────────────────────────────────────────────────────────
x = np.linspace(0, 2, 1000)
t = np.linspace(0, 0.05, 500)
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

Np = 10           # prediction horizon (in steps)
Q = 10000         # state tracking weight
R = 0.0001         # control effort weight
alpha = 0 * Q   # terminal‐state weight
S = 0 * R  # rate‐penalty weight (tune as needed)
v = 20  # laser scan speed (units of x per unit time)

# ─── Laser Discretization parameters ──────────────────────────────────────────────────────────
P_step    = 1             # your minimum power increment [W]
T_min     = 0.001*1/10               # minimum time to hold each step [s]
hold_steps = int(np.ceil(T_min / dt))  # how many MPC ticks that is
last_change_idx = -hold_steps    # so that at k=0 we are “allowed” to change
last_pid_change_idx = -hold_steps    # initialize so at k=0 you can change
loop_times = [] # Store loop times for performance analysis
U_frames = [temp_profile.copy()] # Initialize U_frames and store the initial state at t = 0

def desired_temp(time):
    """
    Complex reference temperature profile:
    - Linear ramp from base to peak and back
    - Gaussian spike at midpoint
    - Sinusoidal ripple throughout
    """
    t_max = np.max(t)
    t_mid = 0.5 * t_max
    base = 1500
    peak = 3000

    # Linear ramp component
    if time <= t_mid:
        ref = base + (peak - base) * (time / t_mid)
    else:
        ref = peak - (peak - base) * ((time - t_mid) / t_mid)

    # Gaussian spike at midpoint
    spike_magnitude = 0.05
    spike_width = 0.2 * t_max
    spike = spike_magnitude * np.exp(-((time - t_mid)**2) / (2 * spike_width**2))

    # Sinusoidal ripple throughout
    ripple_amplitude = 0.0
    ripple_frequency = 4 * np.pi / t_max  # Two cycles over full time
    ripple = ripple_amplitude * np.sin(ripple_frequency * time)

    return ref + spike + ripple

# def desired_temp(time):
#     temp = 2500
#     return temp

# ─── Simulation storage ─────────────────────────────────────────────────────
x_temp = np.array([[np.max(temp_profile)]])  # Use actual initial max temp

# Initialize logs with the true initial condition (so plots start at t=0)
t_log     = [0.0]
x_log     = [np.max(temp_profile)]
u_log     = [0.0]
x_pid_log = [np.max(temp_profile_pid)]
u_pid_log = [0.0]

# Disturbance parameters
distub = 500.0  # Disturbance magnitude
noise_std_dev = 0.0  # Standard deviation for noise in reference signal

# ─── Main simulation loop ──────────────────────────────────────────────────
for k in range(num_time_steps):
    
    start_time = time.time() # Start timing the loop
    

    # Timing variables for the simulation
    t_k = k * dt
    ref = desired_temp(t_k)
    
    # Disturbance: simulate a disturbance at t=1/2t_max
    if k == num_time_steps // 2:
        x_laser = np.clip(v * t_k, x.min(), x.max())
        idx = np.argmin(np.abs(x - x_laser))
        disturbance = distub * np.exp(-((x - x[idx])**2) / (2 * (0.001)**2))  # σ=0.01
        temp_profile += disturbance
        temp_profile_pid += disturbance

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
    
    # initialize best_u to previous control
    best_u, min_cost = u_prev, None 
    # if we are beyond the hold‐time, run the optimizer
    if (k - last_change_idx) >= hold_steps:
        # one‐arg cost that closes over u_prev and ref_seq_full
        def cost_for_u0(u0):
            x_pred = simulate_plant_over_horizon([u0]*Np, temp_profile, t0=t_k)
            x_pred = np.array(x_pred) + np.random.normal(0, noise_std_dev, size=len(x_pred)) # add noise to the predicted temperatures
            J = sum(Q*(x_pred[j] - ref_seq_full[j])**2 for j in range(1, Np+1))
            J += alpha * (x_pred[-1] - ref_seq_full[-1])**2
            J += R * (u0**2)
            J += S * (u0 - u_prev)**2
            return J

        # minimize over u0 ∈ [0, 5000]
        res = minimize_scalar(cost_for_u0, bounds=(0, 50000000), method='bounded')
        best_u, min_cost = res.x, res.fun

        # register the change only if it really changed
        if abs(best_u - u_prev) > 1e-6:
            last_change_idx = k

        print(f"\nChosen u = {best_u:.3f} (cost={min_cost:.4f})")
    else:
        # within hold‐time, reuse previous control without re‐computing cost
        print(f"\nHolding u = {best_u:.3f} (held from step {last_change_idx})")

    #Apply the best control for one Δt step 
    temp_profile, x_temp_val = pde_step(best_u, temp_profile,t0=t_k)
    melt_temp_mpc = sample_melt_pool(temp_profile, x, t_k, v)
    x_log.append(melt_temp_mpc)
    u_log.append(best_u)
    t_log.append(t_k + dt)   # log the time *after* the step
    
    # ─── PID benchmark step ───────────────────────────────────────────────── 
    # constants
    SAT_LOW, SAT_HIGH = 0.0, 50000000.0
    epsilon = 1e-6

    # compute the current error & so on only if you're allowed to change
    if (k - last_pid_change_idx) >= hold_steps:
        current_pid_temp = sample_melt_pool(temp_profile_pid, x, t_k, v)
        current_pid_temp += np.random.normal(loc=0.0, scale=noise_std_dev)

        error = ref - current_pid_temp
        error_int += error * dt
        error_der = (error - prev_error) / dt

        # raw PID
        u_candidate = Kp*error + Ki*error_int + Kd*error_der

        # saturate & anti‐windup
        u_pid_clipped = np.clip(u_candidate, SAT_LOW, SAT_HIGH)
        if u_pid_clipped <= SAT_LOW + epsilon or u_pid_clipped >= SAT_HIGH - epsilon:
            error_int -= error * dt  # undo if we saturated

        u_pid = u_pid_clipped

        # register the change if it really moved
        if abs(u_pid - (u_pid_log[-1] if u_pid_log else 0.0)) > 1e-6:
            last_pid_change_idx = k
    else:
    # hold previous PID output
        u_pid = u_pid_log[-1] if u_pid_log else 0.0

    # plant step
    temp_profile_pid, x_pid_val = pde_step(u_pid, temp_profile_pid,t0=t_k)
    melt_temp_pid = sample_melt_pool(temp_profile_pid, x, t_k, v)
    prev_error = error

    # logging
    u_pid_log.append(u_pid)
    x_pid_log.append(melt_temp_pid)    

    end_time = time.time() # End timing the loop
    loop_times.append(end_time - start_time)  # Store the loop time
    
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
plt.figure(figsize=(10,5))
plt.plot(t_log,    x_log,     label='Melt Pool Temp (MPC)', linewidth=2)
plt.plot(t_log,    x_pid_log, label='Melt Pool Temp (PID)', linewidth=2)
plt.plot(t_log, [desired_temp(tt) for tt in t_log], '--', label='Reference', linewidth=2)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Melt Pool Temperature (\u00b0C)', fontsize=14)
plt.title('MPC vs PID Tracking of Melt Pool Temperature', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "temperature_tracking.svg"))  # ← Save here
plt.show()

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
plt.savefig(os.path.join(save_dir, "control_signal_tracking.svg"))  # ← Save here
plt.show()

# Plot the loop times as a histogram
plt.figure()
plt.hist(loop_times, bins=30, edgecolor='black')  # You can adjust `bins` as needed
plt.xlabel("Time per iteration (s)")
plt.ylabel("Frequency")
plt.title("Control Loop Timing Histogram")
plt.grid(True)
plt.savefig(os.path.join(save_dir, "loop_timing.svg"))
plt.show()

# # ─── Create Temperature Profile GIF ───
# # Now animate the collected frames
# x_start = 0.0  # initial position of the melt pool
# fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
# line_analytical, = ax.plot([], [], label='Finite Difference Solution', color='green', linewidth=2)
# melt_dot, = ax.plot([], [], 'ro', markersize=8, label='Melt Pool')
# ax.set_title('1-D Temperature Profile Temporal Evolution', fontsize=14.5)
# ax.set_xlabel('x', fontsize=16)
# ax.set_ylabel('$u(x,t)$', fontsize=16)
# ax.legend(fontsize=16)
# ax.tick_params(axis='both', which='major', labelsize=16)
# # Set plot limits
# ax.set_xlim(np.min(x), np.max(x))
# ax.set_ylim(np.min(U_frames), np.max(U_frames))
# # Optional: Time text
# text = ax.text(0.02, 0.90, '', transform=ax.transAxes, fontsize=14)
# # Update function with known position
# def update(frame):
#     time_now = frame * dt
#     melt_x = x_start + v * time_now

#     u_frame = U_frames[frame]
#     melt_temp = np.interp(melt_x, x, u_frame)

#     line_analytical.set_data(x, u_frame)
#     melt_dot.set_data([melt_x], [melt_temp])  # Fixed here
#     text.set_text(f"Time: {time_now:.2f} s")

#     return line_analytical, melt_dot, text
# # Create animation from U_frames
# anim = FuncAnimation(fig, update, frames=len(U_frames), blit=True)
# # Save as GIF
# anim.save(os.path.join(save_dir, 'Finite_DifferenceCDT.gif'), dpi=100, writer='pillow', fps=30)

# # ─── Create Temperature Profile GIF ───-───────────────────────────────
# # Compute melt pool temperature over time
# max_temp_over_time = x_log  # Shape: (num_time_steps,)
# time = t[:len(max_temp_over_time)]  # Ensure time matches the length of max_temp_over_time
# # Set up the figure and axis
# fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
# line, = ax.plot([], [], color='red', linewidth=2, label='Melt Pool Temperature')
# ax.set_title('Melt Pool Temperature Over Time (MPC)', fontsize=14.5)
# ax.set_xlabel('Time', fontsize=16)
# ax.set_ylabel('Temperature (\u00b0C)', fontsize=16)
# ax.legend(fontsize=16)
# ax.tick_params(axis='both', which='major', labelsize=16)
# ax.set_xlim(np.min(time), np.max(time))
# ax.set_ylim(np.min(max_temp_over_time), np.max(max_temp_over_time) * 1.05)
# # Create a text label for the current max
# text = ax.text(0.02, 0.85, '', transform=ax.transAxes, fontsize=16)
# # Animation update function
# def update(frame):
#     current_time = time[:frame+1]
#     current_max = max_temp_over_time[:frame+1]
#     line.set_data(current_time, current_max)
#     text.set_text(f'Time = {time[frame]:.2f}s\nMax T = {max_temp_over_time[frame]:.2f}')
#     return line, text
# # Create animation
# anim = FuncAnimation(fig, update, frames=len(time), blit=True)
# fig.tight_layout()  # Ensures padding is correct
# # Save the animation
# anim.save(os.path.join(save_dir, 'Melt_Pool_Over_TimeCDT.gif'), dpi=100, writer='pillow', fps=30)
