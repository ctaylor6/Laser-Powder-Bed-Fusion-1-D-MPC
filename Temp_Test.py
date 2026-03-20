import numpy as np
import matplotlib.pyplot as plt
from carter_STPO import CTD_STPO_nodal_visual, loaded_sol, eta_specify, zeta_specify, cfem_mesh_dict_x, random_init, x, dof_global_t

def main():
    # Initialize the initial condition
    current_condition = random_init  # Start with the initial condition

    # Define initial laser power
    laser_power = 8.0  # Example initial laser power

    # Iterate over time steps
    for time_step in range(dof_global_t):
        # Dynamically update the laser power (example: sinusoidal variation)
        laser_power = 8.0 + 2.0 * np.sin(2 * np.pi * time_step / dof_global_t)  # Example dynamic update

        # Call CTD_STPO_nodal_visual for the current time step
        U0 = CTD_STPO_nodal_visual(
            np.array(loaded_sol['U_x']),
            np.array(loaded_sol['U_t']),
            np.array(loaded_sol['U_eta_s']),
            np.array(loaded_sol['U_zeta_s']),
            eta_specify * laser_power,  # Scale eta_specify by laser power
            zeta_specify,
            time_step
        )

        # Update the current condition for the next time step
        current_condition = U0 + current_condition  # Use the result as the new initial condition

        # Optionally, plot or store the result for this time step
        plt.plot(x, current_condition, label=f"Time Step {time_step} (Power: {laser_power:.2f})")

    # Finalize the plot
    plt.xlabel("x")
    plt.ylabel("TAPSO Prediction")
    plt.title("TAPSO Prediction Over Time Steps with Dynamic Laser Power")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()

