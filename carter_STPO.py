import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['JAX_DISABLE_MOST_FAVORABLE_DEVICE'] = 'true'
import numpy as onp
import jax
import jax.numpy as np
import time
import os,sys
import matplotlib.pyplot as plt

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from src.CFEM_rf import *
from src.generate_mesh import *

GPU_idx = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_idx)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
jax.config.update("jax_enable_x64", True)

onp.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True, precision=4)

loaded_dict = np.load('/Users/cartertaylor/ws/495_Project/STP_td_tapso_ST_control_carter.npz')
loaded_sol = {key: loaded_dict[key] for key in loaded_dict}

mean_P = 0. #power input mean
mean_H = 0. #initial condition mean
n_kl_f = 5
n_kl_i = 40
n_kl_f_apprx = loaded_sol['eigvals_t'].shape[0]
n_kl_i_apprx = loaded_sol['eigvals_i'].shape[0]

nelem_x = 200
nelem_t = 5
nelem_eta = 300
nelem_zeta = 300

elem_type = 'D1LN2N' # 'D1LN2N'

assert loaded_sol['U_x'].shape[1] == nelem_x + 1; assert loaded_sol['U_t'].shape[1] == nelem_t + 1
assert loaded_sol['U_eta_s'].shape[2] == nelem_eta + 1; assert loaded_sol['U_zeta_s'].shape[2] == nelem_zeta + 1

Lx = 1 # length of the domain
Lt = 0.01
Leta = 300
Lzeta = 300
#%%
s_patch_x = 3; s_patch_t = 3; s_patch_eta = 3; s_patch_zeta = 3
p_X = -1; p_t = -1; p_eta = -1; p_zeta = -1
     
alpha_dil_x = 20; alpha_dil_t = 20; alpha_dil_eta = 20; alpha_dil_zeta = 20

Gauss_Num_CFEM = 3

p_dict={0:0, 1:2, 2:3, 3:4} 
mbasis = p_dict[2]  
### body force
# mode 1
@jax.jit
def fun_x(x):
    return 1

@jax.jit
def fun_t(t):
    return 1

@jax.jit
def fun_eta(eta):
    return 1

@jax.jit
def fun_zeta(zeta):
    return 1

nodes_per_elem = int(elem_type[4:-1]) #same for space and parameter for the same type of elements
dim = int(elem_type[1])
elem_dof = nodes_per_elem*dim    

fun_x_vvmap = jax.jit(jax.vmap(jax.vmap(fun_x, in_axes = (0)), in_axes = (0)))
fun_t_vvmap = jax.jit(jax.vmap(jax.vmap(fun_t, in_axes = (0)), in_axes = (0)))
fun_eta_vvmap = jax.jit(jax.vmap(jax.vmap(fun_eta, in_axes = (0)), in_axes = (0)))
fun_zeta_vvmap = jax.jit(jax.vmap(jax.vmap(fun_zeta, in_axes = (0)), in_axes = (0)))

# Problem setting

## Mesh generation
non_uniform_mesh_bool = False
#XY_host: nodal coord; #Elem_nodes_host: element and its nodal id
x, Elem_nodes_x = uniform_mesh_new(Lx, nelem_x)
t_kl_f, Elem_nodes_t_kl_f = uniform_mesh_new(Lt, n_kl_f)
x_kl_i, Elem_nodes_x_kl_i = uniform_mesh_new(Lx, n_kl_i)
t, Elem_nodes_t = uniform_mesh_new(Lt, nelem_t)
eta, Elem_nodes_eta = uniform_mesh_new(Leta, nelem_eta)
eta = eta - Leta/2 #shift nodal value to avoid non-positive parameters

zeta, Elem_nodes_zeta = uniform_mesh_new(Lzeta, nelem_zeta)
zeta = zeta - Lzeta/2 #shift nodal value to avoid non-positive parameters

start_time_org = time.time()
indices_x, indptr_x = get_adj_mat(Elem_nodes_x, nelem_x+1, s_patch_x)
indices_t_kl_f, indptr_t_kl_f = get_adj_mat(Elem_nodes_t_kl_f, n_kl_f+1, s_patch_t)
indices_x_kl_i, indptr_x_kl_i = get_adj_mat(Elem_nodes_x_kl_i, n_kl_i+1, s_patch_x)
indices_t, indptr_t = get_adj_mat(Elem_nodes_t, nelem_t+1, s_patch_t)
indices_eta, indptr_eta = get_adj_mat(Elem_nodes_eta, nelem_eta+1, s_patch_eta)
indices_zeta, indptr_zeta = get_adj_mat(Elem_nodes_zeta, nelem_zeta+1, s_patch_zeta)
print(f"CFEM adj_s matrix took {time.time() - start_time_org:.4f} seconds")

# patch settings for X
d_c_x = Lx/nelem_x; d_c_t = Lt/nelem_t; d_c_eta = Leta/nelem_eta; d_c_zeta = Lzeta/nelem_zeta
d_c_t_kl_f = Lt/n_kl_f; d_c_x_kl_i = Lx/n_kl_i

a_dil_x = alpha_dil_x * d_c_x; a_dil_t = alpha_dil_t * d_c_t; a_dil_eta = alpha_dil_eta * d_c_eta; a_dil_zeta = alpha_dil_zeta * d_c_zeta
a_dil_t_kl_f = alpha_dil_t * d_c_t_kl_f; a_dil_x_kl_i = alpha_dil_x * d_c_x_kl_i

start_time = time.time()
edex_max_x, ndex_max_x = get_dex_max(indices_x, indptr_x, s_patch_x, Elem_nodes_x)
edex_max_t, ndex_max_t = get_dex_max(indices_t, indptr_t, s_patch_t, Elem_nodes_t)
edex_max_eta, ndex_max_eta = get_dex_max(indices_eta, indptr_eta, s_patch_eta, Elem_nodes_eta)
edex_max_zeta, ndex_max_zeta = get_dex_max(indices_zeta, indptr_zeta, s_patch_zeta, Elem_nodes_zeta)
edex_max_t_kl_f, ndex_max_t_kl_f = get_dex_max(indices_t_kl_f, indptr_t_kl_f, s_patch_t, Elem_nodes_t_kl_f)
edex_max_x_kl_i, ndex_max_x_kl_i = get_dex_max(indices_x_kl_i, indptr_x_kl_i, s_patch_x, Elem_nodes_x_kl_i)

print(f'edex_max took {time.time() - start_time:.4f} seconds')

#Solve for TD matrix: identical in all iterations
input_dict = {
    'coor':{'x':x, 't':t, 'eta':eta, 'zeta':zeta},
    'Elem_nodes':{'x':Elem_nodes_x, 't':Elem_nodes_t, 'eta':Elem_nodes_eta, 'zeta':Elem_nodes_zeta},
    'indices':{'x':indices_x, 't':indices_t, 'eta':indices_eta, 'zeta':indices_zeta},
    'indptr':{'x':indptr_x, 't':indptr_t, 'eta':indptr_eta, 'zeta':indptr_zeta},
    'edex_max':{'x':edex_max_x, 't':edex_max_t, 'eta':edex_max_eta, 'zeta':edex_max_zeta},
    'ndex_max':{'x':ndex_max_x, 't':ndex_max_t, 'eta':ndex_max_eta, 'zeta':ndex_max_zeta},
    's_patch':{'x':s_patch_x, 't':s_patch_t, 'eta':s_patch_eta, 'zeta':s_patch_zeta},
    'a_dil':{'x':a_dil_x, 't':a_dil_t, 'eta':a_dil_eta, 'zeta':a_dil_zeta},    
}

shape_fun_dict = get_CTD_shape_fun_dict(input_dict, mbasis, Gauss_Num_CFEM, elem_type, linear_time = False)
N_til_x = shape_fun_dict['x']['N_til']; N_til_t = shape_fun_dict['t']['N_til']
N_til_eta = shape_fun_dict['eta']['N_til']; N_til_zeta = shape_fun_dict['zeta']['N_til']
Grad_N_til_x = shape_fun_dict['x']['Grad_N_til']; Grad_N_til_t = shape_fun_dict['t']['Grad_N_til']; 
Grad_N_til_eta = shape_fun_dict['eta']['Grad_N_til']; Grad_N_til_zeta = shape_fun_dict['zeta']['Grad_N_til']
JxW_x = shape_fun_dict['x']['JxW']; JxW_t = shape_fun_dict['t']['JxW']
JxW_eta = shape_fun_dict['eta']['JxW']; JxW_zeta = shape_fun_dict['zeta']['JxW']
Ele_patch_n_x = shape_fun_dict['x']['Elemental_patch_nodes_st']; Ele_patch_n_t = shape_fun_dict['t']['Elemental_patch_nodes_st']
Ele_patch_n_eta = shape_fun_dict['eta']['Elemental_patch_nodes_st']; Ele_patch_n_zeta = shape_fun_dict['zeta']['Elemental_patch_nodes_st']

dof_global_t = nelem_t + 1
eta = np.array(eta); zeta = np.array(zeta); x = np.array(x); t = np.array(t)
t_kl_f = np.array(t_kl_f); x_kl_i = np.array(x_kl_i)

cfem_mesh_dict_kl_f = get_Gs_inv(t_kl_f, Elem_nodes_t_kl_f, 
                indices_t_kl_f, indptr_t_kl_f, edex_max_t_kl_f, ndex_max_t_kl_f, 
                a_dil_t_kl_f, mbasis)

cfem_mesh_dict_kl_i = get_Gs_inv(x_kl_i, Elem_nodes_x_kl_i,
                indices_x_kl_i, indptr_x_kl_i, edex_max_x_kl_i, ndex_max_x_kl_i, 
                a_dil_x_kl_i, mbasis)

cfem_mesh_dict_eta = get_Gs_inv(eta, Elem_nodes_eta,
                indices_eta, indptr_eta, edex_max_eta, ndex_max_eta, 
                a_dil_eta, mbasis)

cfem_mesh_dict_zeta = get_Gs_inv(zeta, Elem_nodes_zeta,
                indices_zeta, indptr_zeta, edex_max_zeta, ndex_max_zeta, 
                a_dil_zeta, mbasis)

cfem_mesh_dict_x = get_Gs_inv(x, Elem_nodes_x,
                indices_x, indptr_x, edex_max_x, ndex_max_x, 
                a_dil_x, mbasis)

cfem_mesh_dict_t = get_Gs_inv(t, Elem_nodes_t,
                indices_t, indptr_t, edex_max_t, ndex_max_t, 
                a_dil_t, mbasis)


eigvecs_t = loaded_sol['eigvecs_t']; eigvals_t = loaded_sol['eigvals_t']
eigvecs_i = loaded_sol['eigvecs_i']; eigvals_i = loaded_sol['eigvals_i']


# %%
def CFEM_interp(x_input, x_grid, values, cfem_mesh_dict):
    # Interpolate the values at the specified points using CFEM
    x_input = x_input.reshape(-1, 1)
    x_grid = x_grid.reshape(-1)
    values = values.reshape(-1, 1)
    interp = vmap_CFEM_anypt(x_input, x_grid, values, cfem_mesh_dict) # (num_total_points_for_evaluation, 1)
    interp = interp.reshape(-1) # (num_total_points_for_evaluation,)
    return interp

def batch_interp(eta_s, eta, U_eta_s, cfem_mesh_dict):
    '''
    batch interpolation
    '''
    return jax.vmap(jax.vmap(CFEM_interp, in_axes = (None, None, 0, None)), in_axes=(0, None, 0, None))(eta_s, eta, U_eta_s, cfem_mesh_dict)[:,:,0] #(num_kl_terms, num_mode)

def batch_mode_interp(x_input, x_grid, U_x, cfem_mesh_dict):
    '''
    batch interpolation
    '''
    return jax.vmap(CFEM_interp, in_axes = (None, None, 0, None))(x_input, x_grid, U_x, cfem_mesh_dict)[:,:] #(num_mode, num_pts)


def batch_interp_FE(eta_s, eta, U_eta_s):
    '''
    batch interpolation
    '''
    return jax.vmap(jax.vmap(np.interp, in_axes = (None, None, 0)), in_axes=(0, None, 0))(eta_s, eta.reshape(-1), U_eta_s)#(num_kl_terms, num_mode)


@jax.jit
def CTD_STPO_nodal_visual(U_x, U_t, U_eta_s, U_zeta_s, eta_s, zeta_s, id_t): #u(x) 
    '''
    generate 3d (x,y,z) data for visualization
    dim:(num_ele_x*num_quad, num_ele_y*num_quad, num_ele_z*num_quad)
    
    '''
    t_interpolated = U_t[:, id_t] #dim (num_mode)
    # U = np.sum(U_x[:, :] * np.prod(batch_interp_FE(eta_s[-n_kl_f:], eta, U_eta_s), axis = 0)[:, None], axis = 0) #dim(num_points_x)
    
    U = np.sum(batch_mode_interp(x, x, U_x, cfem_mesh_dict_x) * t_interpolated[:, None] * \
               np.prod(batch_interp(eta_s[:n_kl_f], eta, U_eta_s, cfem_mesh_dict_eta), axis = 0)[:, None] * \
               np.prod(batch_interp(zeta_s[:n_kl_i], zeta, U_zeta_s, cfem_mesh_dict_zeta), axis = 0)[:, None], axis = 0) #dim(num_points_x)
    return U  #dim(num_points_x)

### the above code you don't have to change, the shape functions and interpolation functions are already defined, you just need to use them
# you need to only run them once for all. then change the following code to generate the initial condition and the power series coefficients

#Example arbitrary power series
t_kl_f = t_kl_f.reshape(-1)
p0 = 8 * np.sin(np.pi * t_kl_f) + mean_P - mean_P  # Example arbitrary realization #change this to your case
# Project u_0(x) onto the KL basis
kl_coefficients_t = eigvecs_t.T @ p0 #np.linalg.inv(eigvecs) @ f0
eta_specify = kl_coefficients_t[:]
print(min(eta_specify), max(eta_specify))

#Example arbitrary initial condition
x_kl_i = x_kl_i.reshape(-1)
f0 = 8 * np.sin(np.pi * x_kl_i) + mean_H - mean_H  # Example arbitrary realization #change this to your case
kl_coefficients = eigvecs_i.T @ f0 #np.linalg.inv(eigvecs) @ f0
zeta_specify_no_scale = kl_coefficients[:]
zeta_specify = zeta_specify_no_scale/eigvals_i
force_fun = (eigvecs_i[:, :] * eigvals_i[None, :]) @ (zeta_specify)

# Plot the original realization and KL approximation
#plt.figure(figsize=(12, 6))
# Plot the original realization
#plt.plot(x_kl_i, f0, label="Original Realization", linestyle='--', color='blue')
# Plot the KL approximation
kl_approximation = (eigvecs_i[:, :] * eigvals_i[None, :]) @ zeta_specify
# plt.plot(x_kl_i, kl_approximation, label="KL Approximation", linestyle='-', color='red')

# plt.xlabel("x")
# plt.ylabel("Value")
# plt.title("Original Realization vs KL Approximation")
# plt.legend()
# plt.grid(True)


random_init  = CFEM_rf_interp(x.reshape(-1), x_kl_i.reshape(-1), force_fun.reshape(-1), eigvecs_i, n_kl_f, cfem_mesh_dict_kl_i) + mean_H

# #tapso prediction
# U_tapso = np.zeros((len(x), dof_global_t))  # Initialize TAPSO prediction array
# for time_step in range(dof_global_t):
#     # Perform TAPSO update
#     U0 = CTD_STPO_nodal_visual(np.array(loaded_sol['U_x']), np.array(loaded_sol['U_t']), 
#                                np.array(loaded_sol['U_eta_s']), np.array(loaded_sol['U_zeta_s']), eta_specify, zeta_specify, time_step) 
#     U_tapso = U_tapso.at[:, time_step].set(U0 + random_init)   # Store the TAPSO prediction
    
# # Plot the TAPSO prediction
# plt.figure(figsize=(12, 6))
# for time_step in range(dof_global_t):
#     plt.plot(x, U_tapso[:, time_step], label=f"Time Step {time_step}")
# plt.xlabel("x")
# plt.ylabel("TAPSO Prediction")
# plt.title("TAPSO Prediction Over Time Steps")
# plt.legend()
# plt.grid(True)
# plt.show()

# Initialize the initial condition for discrete time TAPSO
current_condition = random_init  # Start with the initial condition

# Define initial laser power
laser_power = 8.0  # Example initial laser power

# Iterate over time steps
for time_step in range(dof_global_t):
    # Dynamically update the laser power (example: sinusoidal variation)
    laser_power = 8.0 + 2.0 * np.sin(2 * np.pi * time_step / dof_global_t)  # Example dynamic update

    # Perform TAPSO update for the current time step
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
    # plt.plot(x, current_condition, label=f"Time Step {time_step} (Power: {laser_power:.2f})")

# # Finalize the plot
# plt.xlabel("x")
# plt.ylabel("TAPSO Prediction")
# plt.title("TAPSO Prediction Over Time Steps with Dynamic Laser Power")
# plt.legend()
# plt.grid(True)
# plt.show()