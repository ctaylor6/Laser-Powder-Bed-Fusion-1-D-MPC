###transient heat transfer code with Tensor Decomposition Formulation
from matplotlib.pyplot import axis
import numpy as onp
from scipy.sparse import csr_matrix
import jax
import jax.numpy as np
import jax.experimental.sparse as jsp
from jax import lax
from jax.experimental.sparse import BCOO, BCSR
# import numpy as cp
# import scipy.sparse as sparse
# import scipy.sparse.linalg as splinalg
from functools import partial, total_ordering
from scipy.io import savemat
import os,sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
rundir = os.path.join(parentdir, 'run')
sys.path.append(rundir)

from scipy.sparse.linalg import spilu, splu, spsolve, inv
from src.CFEM_shape_fun import *
from src.generate_mesh import *
import time

#added convection bc terms and verified

#jl.seval("using MatrixEquations")

onp.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True, precision=7)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# TODO: change to C-HiDeNN interpolation

def solve_n_kl_terms(eigvals, tol = 1e-5):
    """determin the number of KL terms to use in the approximation

    Args:
        eigvals (num_discretization): number of discretization points; descending order
    """
    eigvals_rescale = eigvals/ eigvals[0] # rescale the eigenvalues according to the largest one
    trimmed = eigvals_rescale[eigvals_rescale > tol]
    
    n_kl_terms = trimmed.shape[0]
    return n_kl_terms
    
def FE_eigvec_1D(x, x_I, C_I):
    """_summary_

    Args:
        x (num_total_points_for_evaluation): gauss points
        x_I (num_sample_in_x): mesh nodal points
        C_I (num_sample_in_x): mesh nodal values

    Returns:
        interpolated_fun@gauss points (num_total_points_for_evaluation): (num_total_points_for_evaluation)
    """
    return jnp.interp(x, x_I, C_I)

    
def FE_eigvecs(x, x_I, C_Ik):
    fun = jax.vmap(FE_eigvec_1D, in_axes=(None, None, 1)) #(num_kl, num_sample)
    return np.transpose(fun(x, x_I, C_Ik), (1, 0)) #(num_sample, num_kl,)



def FE_rf_interp(x_pred, x0, u0, eigvecs, n_kl_terms):
    """return values of the KL approxiaed function at any x_pred points
        #basis @ coefficients
    Args:
        x_gauss (num_total_points_for_evaluation): _description_
        x0 (num_sample): original sampling points for x
        u0 (num_sample): original sampling points for realization of u
        eigvecs (num_sample, num_sample): eigenvectors of the covariance matrix
        n_kl_terms (1): number of KL terms to use in the approximation

    Returns:
        approx_u: (num_total_points_for_evaluation) approximated realization of u
    """
    kl_coefficients = eigvecs.T @ u0  # Compute coefficients for the KL expansion
    eigvecs = FE_eigvecs(x_pred, x0, eigvecs) #(num_sample, num_sample)
    approx_u0 = eigvecs[:, :n_kl_terms] @ kl_coefficients[:n_kl_terms]
    return approx_u0


def FE_rf_basis(x_gauss, x0, eigvecs, n_kl_terms):
    """return values of the KL approxiaed function basis at the gauss points

    Args:
        x_gauss (num_total_points_for_evaluation): _description_
        x0 (num_sample): original sampling points for x
        eigvecs (num_sample, num_sample): eigenvectors of the covariance matrix

    Returns:
        approx_u: 
        basis: [num_total_points_for_evaluation, num_sample]
        rf_basis: [KL_terms, num_total_points_for_evaluation]: reduced KL basis
    """
    full_basis = FE_eigvecs(x_gauss, x0, eigvecs) #( num_sample, num_kl,)
    rf_basis = (full_basis[:, :n_kl_terms]).T  # (num_kl, num_sample)
    return rf_basis


def get_CTD_shape_fun(x, t, ksi,
                  Elem_nodes_x,  Elem_nodes_t, Elem_nodes_ksi,
                  indices_x, indptr_x, indices_t, indptr_t,  indices_ksi, indptr_ksi,
                  edex_max_x, ndex_max_x, edex_max_t, ndex_max_t, edex_max_ksi, ndex_max_ksi,
                  s_patch_x, s_patch_t, s_patch_ksi, a_dil_x, a_dil_t, a_dil_ksi, mbasis, 
                  Gauss_Num_CFEM, elem_type):
    
    radial_basis = 'cubicSpline'
    nelem_x = len(Elem_nodes_x)
    nelem_t = len(Elem_nodes_t)
    nelem_ksi = len(Elem_nodes_ksi)
    nodes_per_elem = 2; dim = 1
    shape_vals = get_shape_vals(Gauss_Num_CFEM, dim, elem_type) # (quad_num, nodes_per_elem)  --shape fun values @ quads: same for differnt parameters

    #Need this Jacobian info for different parameters for integration
    # N_til_X: (nelem, quad_num, edex_max)
    (N_til_x, Grad_N_til_x, JxW_x, 
     Elemental_patch_nodes_st_x) = get_CFEM_shape_fun(onp.arange(nelem_x), nelem_x,
               x, Elem_nodes_x, shape_vals, Gauss_Num_CFEM, dim, elem_type, nodes_per_elem,
               indices_x, indptr_x, s_patch_x, edex_max_x, ndex_max_x, a_dil_x, mbasis, radial_basis)

    (N_til_t, Grad_N_til_t, JxW_t, 
     Elemental_patch_nodes_st_t) = get_CFEM_shape_fun(onp.arange(nelem_t), nelem_t,
               t, Elem_nodes_t, shape_vals, Gauss_Num_CFEM, dim, elem_type, nodes_per_elem,
               indices_t, indptr_t, s_patch_t, edex_max_t, ndex_max_t, a_dil_t, mbasis, radial_basis)
     
    (N_til_ksi, Grad_N_til_ksi, JxW_ksi,
        Elemental_patch_nodes_st_ksi) = get_CFEM_shape_fun(onp.arange(nelem_ksi), nelem_ksi,
                ksi, Elem_nodes_ksi, shape_vals, Gauss_Num_CFEM, dim, elem_type, nodes_per_elem,
                indices_ksi, indptr_ksi, s_patch_ksi, edex_max_ksi, ndex_max_ksi, a_dil_ksi, mbasis, radial_basis) 
     
     
    return (N_til_x, N_til_t, N_til_ksi, 
           Grad_N_til_x, Grad_N_til_t, Grad_N_til_ksi, 
           JxW_x, JxW_t, JxW_ksi, 
           Elemental_patch_nodes_st_x, Elemental_patch_nodes_st_t, Elemental_patch_nodes_st_ksi)


def get_CTD_shape_fun_dict(input_dict, mbasis, Gauss_Num_CFEM, elem_type, linear_time = True):
    # Extract coordinate dictionary
    coor = input_dict['coor']  # {'x': ..., 't': ..., 'ksi': ...}

    # Extract element nodes dictionary
    elem_nodes = input_dict['Elem_nodes']  # {'x': ..., 't': ..., 'ksi': ...}

    # Extract indices dictionary
    indices_ = input_dict['indices']  # {'x': ..., 't': ..., 'ksi': ...}

    # Extract indptr dictionary
    indptr_ = input_dict['indptr']  # {'x': ..., 't': ..., 'ksi': ...}

    # Extract max indices dictionary
    edex_max_ = input_dict['edex_max']  
    ndex_max_ = input_dict['ndex_max']
    # Extract additional parameters required for computation
    patch_sizes = input_dict['s_patch']  # {'x': s_patch_x, 't': s_patch_t, 'ksi': s_patch_ksi}
    a_dil_ = input_dict['a_dil']  # {'x': a_dil_x, 't': a_dil_t, 'ksi': a_dil_ksi}
    
    radial_basis = 'cubicSpline'
    nodes_per_elem = 2
    dim = 1
    shape_vals = get_shape_vals(Gauss_Num_CFEM, dim, elem_type)

    results = {}

    # Loop over coordinate types ('x', 't', 'ksi') to perform computations
    for coord in coor:
        
        nelem_coords = len(elem_nodes[coord])
        
        # Fetch parameters from the dictionaries
        value = coor[coord]
        elem_node = elem_nodes[coord]
        indices = indices_[coord]
        indptr = indptr_[coord]
        s_patch = patch_sizes[coord]
        edex_max = edex_max_[coord]
        ndex_max = ndex_max_[coord]
        a_dil = a_dil_[coord]

        # Perform shape function computation
        (N_til, Grad_N_til, JxW, 
         Elemental_patch_nodes_st) = get_CFEM_shape_fun(onp.arange(nelem_coords), nelem_coords,
                   value, elem_node, shape_vals, Gauss_Num_CFEM, dim, elem_type, nodes_per_elem,
                   indices, indptr, s_patch, edex_max, ndex_max, a_dil, mbasis, radial_basis)

        results[coord] = {
            'N_til': N_til,
            'Grad_N_til': Grad_N_til,
            'JxW': JxW,
            'Elemental_patch_nodes_st': Elemental_patch_nodes_st
        }

        # Store results in a dictionary
    if linear_time:
        # Fetch parameters from the dictionaries
        value = coor['t']
        elem_node = elem_nodes['t']
        Grad_N_t, JxW_t = get_shape_grads(Gauss_Num_CFEM, dim, elem_type, value, elem_node) # (nelem, quad_num, nodes_per_elem, dim)
        
        results['t'] = {
            'N_til': shape_vals,
            'Grad_N_til': Grad_N_t,
            'JxW': JxW_t,
            'Elemental_patch_nodes_st': elem_node
        }
    return results

def CFEM_eigvec_1D(x_gauss, grid, C_I, cfem_mesh_dict):
    """_summary_

    Args:
        x_gauss (num_total_points_for_evaluation): gauss points
        grid: (num_grid)
        C_I (num_sample_in_x): mesh nodal values

    Returns:
        interpolated_fun@gauss points (num_total_points_for_evaluation): _description_
    """
    x_input = x_gauss.reshape(-1, 1)
    grid = grid.reshape(-1)
    values = C_I.reshape(-1, 1)
    C_gauss = vmap_CFEM_anypt(x_input, grid, values, cfem_mesh_dict) # (num_total_points_for_evaluation, 1)
    C_gauss = C_gauss.reshape(-1)
    return C_gauss

vmap_CFEM_eigvec_1D = jax.vmap(CFEM_eigvec_1D, in_axes=(None, None, 1, None)) #(num_kl, num_sample)

def CFEM_eigvecs(x_gauss, grid, C_Ik, cfem_mesh_dict):
    KL_interpolation = vmap_CFEM_eigvec_1D(x_gauss, grid, C_Ik, cfem_mesh_dict) #(num_kl, num_sample)
    return np.transpose(KL_interpolation, (1, 0)) #( num_sample, num_kl,)


def CFEM_rf_basis(x_gauss, grid, C_Ik, n_kl_terms, cfem_mesh_dict):
    """_summary_

    Args:
        x_gauss (num_total_points_for_evaluation): _description_
        x0 (num_sample): original sampling points for x
        eigvecs (num_sample, num_sample): eigenvectors of the covariance matrix

    Returns:
        approx_u: 
        basis: [num_total_points_for_evaluation, num_sample]
        rf_basis: [KL_terms, num_total_points_for_evaluation]: reduced KL basis
    """
    full_basis = CFEM_eigvecs(x_gauss, grid, C_Ik, cfem_mesh_dict) #( num_sample, num_kl,)
    rf_basis = (full_basis[:, :n_kl_terms]).T  # (num_kl_select, num_sample)
    return rf_basis

def CFEM_rf_interp(x_pred, x0, u0, eigvecs, n_kl_terms, cfem_mesh_dict):
    """return values of the KL approxiaed function at any x_pred points
        only for 1D random field
        #basis @ coefficients
    Args:
        x_pred (num_total_points_for_evaluation): _description_
        x0 (num_sample): original sampling points for x
        u0 (num_sample): original sampling points for realization of u
        eigvecs (num_sample, num_sample): eigenvectors of the covariance matrix
        n_kl_terms (1): number of KL terms to use in the approximation

    Returns:
        approx_u: (num_total_points_for_evaluation) approximated realization of u
    """
    kl_coefficients = eigvecs.T @ u0  # Compute coefficients for the KL expansion
    eigvecs = CFEM_eigvecs(x_pred, x0, eigvecs, cfem_mesh_dict) #(num_sample, num_sample)
    approx_u0 = eigvecs[:, :n_kl_terms] @ kl_coefficients[:n_kl_terms]
    return approx_u0


def CFEM_eigvec_der_1D(x_gauss, grid, C_I, cfem_mesh_dict):
    """_summary_

    Args:
        x_gauss (num_total_points_for_evaluation): gauss points
        grid: (num_grid)
        C_I (num_sample_in_x): mesh nodal values

    Returns:
        interpolated_fun@gauss points (num_total_points_for_evaluation): _description_
    """
    x_input = x_gauss.reshape(-1, 1)
    grid = grid.reshape(-1)
    values = C_I.reshape(-1, 1)
    C_gauss = vmap_CFEM_der_anypt(x_input, grid, values, cfem_mesh_dict) # (num_total_points_for_evaluation, 1)
    C_gauss = C_gauss.reshape(-1)
    return C_gauss

vmap_CFEM_eigvec_der_1D = jax.vmap(CFEM_eigvec_der_1D, in_axes=(None, None, 1, None)) #(num_kl, num_sample)

def CFEM_eigvecs_der(x_gauss, grid, C_Ik, cfem_mesh_dict):
    KL_interpolation = vmap_CFEM_eigvec_der_1D(x_gauss, grid, C_Ik, cfem_mesh_dict) #(num_kl, num_sample)
    return np.transpose(KL_interpolation, (1, 0)) #( num_sample, num_kl,)


def CFEM_rf_der_basis(x_gauss, grid, C_Ik, n_kl_terms, cfem_mesh_dict):
    """_summary_

    Args:
        x_gauss (num_total_points_for_evaluation): _description_
        x0 (num_sample): original sampling points for x
        eigvecs (num_sample, num_sample): eigenvectors of the covariance matrix

    Returns:
        approx_u: 
        basis: [num_total_points_for_evaluation, num_sample]
        rf_basis: [KL_terms, num_total_points_for_evaluation]: reduced KL basis
    """
    full_basis = CFEM_eigvecs_der(x_gauss, grid, C_Ik, cfem_mesh_dict) #( num_sample, num_kl,)
    rf_basis = (full_basis[:, :n_kl_terms]).T  # (num_kl_select, num_sample)
    return rf_basis

def CFEM_rf_der_interp(x_pred, x0, u0, eigvecs, n_kl_terms, cfem_mesh_dict):
    """return gradient values of the KL approxiaed function at any x_pred points
        only for 1D random field
        #basis @ coefficients
    Args:
        x_pred (num_total_points_for_evaluation): _description_
        x0 (num_sample): original sampling points for x
        u0 (num_sample): original sampling points for realization of u
        eigvecs (num_sample, num_sample): eigenvectors of the covariance matrix
        n_kl_terms (1): number of KL terms to use in the approximation

    Returns:
        approx_u: (num_total_points_for_evaluation) approximated realization of u
    """
    kl_coefficients = eigvecs.T @ u0  # Compute coefficients for the KL expansion
    eigvecs = CFEM_eigvecs_der(x_pred, x0, eigvecs, cfem_mesh_dict) #(num_sample, num_sample)
    approx_u0 = eigvecs[:, :n_kl_terms] @ kl_coefficients[:n_kl_terms]
    return approx_u0