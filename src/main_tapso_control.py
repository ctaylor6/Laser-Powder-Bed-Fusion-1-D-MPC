#!/usr/bin/env python3 env called jax-cpu-env
import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['JAX_DISABLE_MOST_FAVORABLE_DEVICE'] = 'true'
import numpy as np
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

from carter_STPO import (
    CTD_STPO_nodal_visual,
    loaded_sol,
    eta_specify,
    zeta_specify,
    random_init,
    x,
    dof_global_t
)

# New: save directory for plots
save_dir = "TAPSO Plots"
os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

# MPC parameters
Np = 5                # horizon length
Q = 0              # state tracking weight
R = 0.000000               # control effort weight
S = 0 * R            # rate‑penalty weight
alpha = 0 * Q         # terminal‑state weight
noise_std_dev = 0.0  # standard deviation for prediction noise
P_max = 10000           # max laser power
dt = 1                 # time step size

# ─── TAPSO plant step ─────────────────────────────────────────────────────────
def tapso_step(current_profile, laser_power, step_idx):
    # Heat with actual power
    U_actual = CTD_STPO_nodal_visual(
        np.array(loaded_sol['U_x']),
        np.array(loaded_sol['U_t']),
        np.array(loaded_sol['U_eta_s']),
        np.array(loaded_sol['U_zeta_s']),
        eta_specify * laser_power,
        zeta_specify,
        step_idx
    )

    # Heat when power = 0
    U_passive = CTD_STPO_nodal_visual(
        np.array(loaded_sol['U_x']),
        np.array(loaded_sol['U_t']),
        np.array(loaded_sol['U_eta_s']),
        np.array(loaded_sol['U_zeta_s']),
        eta_specify * 0.0,
        zeta_specify,
        step_idx
    )

    # Subtract passive drift so u=0 means no effect
    return current_profile + (U_actual - U_passive) 

# ─── TAPSO plant step ─────────────────────────────────────────────────────────
def simulate_tapso_over_horizon(u_seq, init_profile, start_step):
    temps = [init_profile.max()]
    profile = init_profile.copy()
    for j, u_j in enumerate(u_seq):
        profile = tapso_step(profile, u_j, start_step + j)
        temps.append(profile.max())
    return temps

# ─── 2) MPC cost (takes 7 args: u0, current_profile, step_idx, ref_seq, u_prev, noise, Np) ───
def cost_for_power(u0, current_profile, step_idx, ref_seq_full, u_prev, noise_std_dev, Np):
    # Predict future temperature evolution under constant control u0
    x_pred = []
    state = current_profile.copy()
    for j in range(1, Np + 1):
        state = tapso_step(state, u0, step_idx + j)
        x_pred.append(state.max())
    
    x_pred = np.array(x_pred) + np.random.normal(0, noise_std_dev, size=Np)

    # Cost = tracking error + penalties
    J = sum(Q * (x_pred[j] - ref_seq_full[j+1])**2 for j in range(Np))  # match time alignment
    J += alpha * (x_pred[-1] - ref_seq_full[-1])**2
    J += R * (u0**2)
    J += S * (u0 - u_prev)**2
    return J

# desired trajectory (could be a function of time)
def reference_temperature(k):
    return 0  # constant setpoint, for example

# ─── Main, with 1‑step MPC loop ───────────────────────────────────────────────
def main():
    # clear screen
    os.system("cls" if os.name=="nt" else "clear")

    # initial TAPSO condition
    state = random_init.copy()

    # logs
    t_log, Tmax_log, u_log = [0.0], [state.max()], [0.0]
    frames = []

    for k in range(dof_global_t):
        # current reference
        T_ref = reference_temperature(k)

        t_k = k * dt
        # Precompute reference trajectory for this horizon
        ref_seq_full = [reference_temperature(t_k + j*dt) for j in range(Np+1)]
        u_prev = u_log[-1] if u_log else 0.0

        res = minimize_scalar(
            lambda u: cost_for_power(
                u,
                state,
                k,
                ref_seq_full,
                u_prev,
                noise_std_dev,
                Np         # now matches the 7th parameter
            ),
            bounds=(0, P_max),
            method="bounded"
        )
        u_opt = res.x

        # apply optimal power
        state = tapso_step(state, u_opt, k)

        # log
        t_log.append(k+1)
        Tmax_log.append(state.max())
        u_log.append(u_opt)
        frames.append(state.copy())

        print(f"[Step {k:3d}] P* = {u_opt:7.1f}, Tmax = {state.max():.4f}, ref = {T_ref:.4f}")

    # ─── Plots ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(2,1,figsize=(8,6), sharex=True)
    ax[0].plot(t_log, Tmax_log, "-o", label="Tmax")
    ax[0].plot(t_log, [reference_temperature(k) for k in t_log], "--", label="ref")
    ax[0].set_ylabel("Max Temp")
    ax[0].legend()
    ax[1].plot(t_log, u_log, "-o", label="Power")
    ax[1].set_ylabel("Laser Power")
    ax[1].set_xlabel("Time Step")
    ax[1].legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(save_dir, "tapso_mpc_tracking.svg"))  # ← Save here
    
    # # ─── Simple GIF ───────────────────────────────────────────────────────────
    # fig2, ax2 = plt.subplots(figsize=(8,3))
    # ax2.set_xlim(x.min(), x.max())
    # all_vals = np.vstack(frames)
    # ax2.set_ylim(all_vals.min(), all_vals.max())
    # line, = ax2.plot([], [], lw=2)
    # txt = ax2.text(0.02, 0.9, "", transform=ax2.transAxes)

    # def update(i):
    #     line.set_data(x, frames[i])
    #     txt.set_text(f"Step {i}, P={u_log[i]:.1f}")
    #     return line, txt

    # anim = matplotlib.animation.FuncAnimation(
    #     fig2, update, frames=len(frames), blit=True, interval=100
    # )
    # anim.save("tapso_mpc.gif", writer="pillow", fps=10)

    # print("Done: tapso_mpc_tracking.png, tapso_mpc.gif")

if __name__ == "__main__":
    main()
