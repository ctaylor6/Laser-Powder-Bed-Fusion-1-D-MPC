import numpy as onp
from scipy.sparse import csc_matrix
import jax
import jax.numpy as np
from jax.experimental.sparse import BCOO
from itertools import combinations
from jax.experimental.sparse import BCSR
from functools import partial
import os,sys
# petsc4py.init()


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # add this for memory pre-allocation
jax.config.update("jax_enable_x64", True)


def GaussSet(Gauss_Num = 2, cuda=False):
    if Gauss_Num == 2:
        Gauss_Weight1D = [1, 1]
        Gauss_Point1D = [-1/np.sqrt(3), 1/np.sqrt(3)]
    
    elif Gauss_Num == 0:
        Gauss_Weight1D = [1.]
        Gauss_Point1D = [-1.]
       
    elif Gauss_Num == 3:
        Gauss_Weight1D = [0.55555556, 0.88888889, 0.55555556]
        Gauss_Point1D = [-0.7745966, 0, 0.7745966]
       
        
    elif Gauss_Num == 4:
        Gauss_Weight1D = [0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538]
        Gauss_Point1D = [-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526]

    elif Gauss_Num == 6: # double checked, 16 digits
        Gauss_Weight1D = [0.1713244923791704, 0.3607615730481386, 0.4679139345726910, 
                          0.4679139345726910, 0.3607615730481386, 0.1713244923791704]
        Gauss_Point1D = [-0.9324695142031521, -0.6612093864662645, -0.2386191860831969, 
                         0.2386191860831969, 0.6612093864662645, 0.9324695142031521]

       
    elif Gauss_Num == 8: # double checked, 20 digits
        Gauss_Weight1D=[0.10122853629037625915, 0.22238103445337447054, 0.31370664587788728733, 0.36268378337836198296,
                        0.36268378337836198296, 0.31370664587788728733, 0.22238103445337447054,0.10122853629037625915]
        Gauss_Point1D=[-0.960289856497536231684, -0.796666477413626739592,-0.525532409916328985818, -0.183434642495649804939,
                        0.183434642495649804939,  0.525532409916328985818, 0.796666477413626739592,  0.960289856497536231684]
        
    elif Gauss_Num == 10:
        Gauss_Weight1D=[0.0666713443086881, 0.1494513491505806, 0.2190863625159820, 0.2692667193099963, 0.2955242247147529,
                        0.2955242247147529, 0.2692667193099963, 0.2190863625159820, 0.1494513491505806, 0.0666713443086881]
        Gauss_Point1D=[-0.9739065285171717, -0.8650633666889845, -0.6794095682990244, -0.4333953941292472, -0.1488743389816312,  
                        0.1488743389816312,  0.4333953941292472,  0.6794095682990244,  0.8650633666889845,  0.9739065285171717]
        
    elif Gauss_Num == 20:
        Gauss_Weight1D=[0.017614007, 0.04060143, 0.062672048, 0.083276742,0.10193012, 0.118194532,0.131688638,
                        0.142096109, 0.149172986, 0.152753387,0.152753387,0.149172986, 0.142096109, 0.131688638,
                        0.118194532,0.10193012, 0.083276742,0.062672048,0.04060143,0.017614007]
            
        Gauss_Point1D=[-0.993128599, -0.963971927, -0.912234428, -0.839116972, -0.746331906, -0.636053681,
                        -0.510867002, -0.373706089, -0.227785851, -0.076526521, 0.076526521, 0.227785851,
                        0.373706089, 0.510867002, 0.636053681, 0.746331906, 0.839116972, 0.912234428, 0.963971927, 0.993128599]
    
    return Gauss_Weight1D, Gauss_Point1D


def get_quad_points(Gauss_Num, dim):
    """ Quadrature point and weight generator
    --- Inputs ---
    --- Outputs ---
    """
    Gauss_Weight1D, Gauss_Point1D = GaussSet(Gauss_Num)
    quad_points, quad_weights = [], []
    
    for ipoint, iweight in zip(Gauss_Point1D, Gauss_Weight1D):
        if dim == 1:
            quad_points.append([ipoint])
            quad_weights.append(iweight)
        else:
            for jpoint, jweight in zip(Gauss_Point1D, Gauss_Weight1D):
                if dim == 2:
                    quad_points.append([ipoint, jpoint])
                    quad_weights.append(iweight * jweight)
                else: # dim == 3
                    for kpoint, kweight in zip(Gauss_Point1D, Gauss_Weight1D):
                        quad_points.append([ipoint, jpoint, kpoint])
                        quad_weights.append(iweight * jweight * kweight)
    
    quad_points = np.array(quad_points) # (quad_degree*dim, dim)
    quad_weights = np.array(quad_weights) # (quad_degree,)
    return quad_points, quad_weights


def get_shape_val_functions(elem_type):
    """ Shape function generator
    """
    ############ 1D ##################
    if elem_type == 'D1LN2N': # 1D linear element
        f1 = lambda x: 1./2.*(1 - x[0])
        f2 = lambda x: 1./2.*(1 + x[0]) 
        shape_fun = [f1, f2] # a list of functions
    
    return shape_fun

def get_shape_grad_functions(elem_type):
    """ Shape function gradient in the parent domain
    """
    shape_fns = get_shape_val_functions(elem_type)
    return [jax.grad(f) for f in shape_fns]

#@partial(jax.jit, static_argnames=['Gauss_Num', 'dim', 'elem_type']) # necessary
def get_shape_vals(Gauss_Num, dim, elem_type):
    """ Measure shape function values at quadrature points
    """
    shape_val_fns = get_shape_val_functions(elem_type)
    quad_points, quad_weights = get_quad_points(Gauss_Num, dim)
    shape_vals = []
    for quad_point in quad_points:
        physical_shape_vals = []
        for shape_val_fn in shape_val_fns:
            physical_shape_val = shape_val_fn(quad_point) 
            physical_shape_vals.append(physical_shape_val)
 
        shape_vals.append(physical_shape_vals)

    shape_vals = np.array(shape_vals) # (quad_num, nodes_per_elem)
    return shape_vals #N_I at different quads

def get_shape_vals_at_pt(E_coor, dim, elem_type):
    """ Measure shape function values at any point
    """    
    shape_val_fns = get_shape_val_functions(elem_type)
@partial(jax.jit, static_argnames=['Gauss_Num', 'dim', 'elem_type']) # necessary
def get_shape_grads(Gauss_Num, dim, elem_type, XY, Elem_nodes):
    """ Meature shape function gradient values at quadrature points
    --- Outputs
    shape_grads_physical: shape function gradient in physcial coordinate (nelem, quad_num, nodes_per_elem, dim)
    JxW: Jacobian determinant times Gauss quadrature weights (nelem, quad_num)
    """
    shape_grad_fns = get_shape_grad_functions(elem_type)
    quad_points, quad_weights = get_quad_points(Gauss_Num, dim)
    shape_grads = []
    for quad_point in quad_points:
        physical_shape_grads = []
        for shape_grad_fn in shape_grad_fns:
            physical_shape_grad = shape_grad_fn(quad_point)
            physical_shape_grads.append(physical_shape_grad)
        shape_grads.append(physical_shape_grads)

    shape_grads = np.array(shape_grads) # (quad_num, nodes_per_elem, dim)
    physical_coos = np.take(XY, Elem_nodes, axis=0) # (nelem, nodes_per_elem, dim)
    jacobian_dx_deta = np.sum(physical_coos[:, None, :, :, None] * shape_grads[None, :, :, None, :], axis=2, keepdims=True) # dx/deta
    # (nelem, quad_num, nodes_per_elem, dim, dim) -> (nelem, quad_num, 1, dim, dim)
    
    jacbian_det = np.squeeze(np.linalg.det(jacobian_dx_deta)) # det(J) (nelem, quad_num)
    jacobian_deta_dx = np.linalg.inv(jacobian_dx_deta) # deta/dx (nelem, quad_num, 1, dim, dim)
    shape_grads_physical = (shape_grads[None, :, :, None, :] @ jacobian_deta_dx)[:, :, :, 0, :]
    JxW = jacbian_det * quad_weights[None, :] #(nelem, quad_num)
    return shape_grads_physical, JxW 


def linear_shape_function(x, nodes):
    """
    Compute the linear shape functions for a given position x and element nodes.

    Args:
        x (float): The position where the shape functions are evaluated.
        nodes (ndarray): Array of shape (2,) containing the element nodes.

    Returns:
        ndarray: Array of shape (2,) containing the computed shape functions.
    """
    N = np.zeros(2)
    N = N.at[0].set((nodes[1] - x) / (nodes[1] - nodes[0]))
    N = N.at[1].set((x - nodes[0]) / (nodes[1] - nodes[0]))
    return N

def CHiDeNN_linear_shape_function(x, nodes):
    """
    Compute the linear shape functions for a given position x and element nodes.

    Args:
        x (float): The position where the shape functions are evaluated.
        nodes (ndarray): Array of shape (2,) containing the element nodes.

    Returns:
        ndarray: Array of shape (2,) containing the computed shape functions.
    """
    N = np.zeros(2)
    N = N.at[0].set((nodes[1] - x) / (nodes[1] - nodes[0]))
    N = N.at[1].set((x - nodes[0]) / (nodes[1] - nodes[0]))
    return N

def find_element(interpolation_point, element_nodes):
    """
    Find the element that contains the given interpolation point.

    Args:
        interpolation_point (float): The interpolation point.
        element_nodes (ndarray): Array of shape (num_elements, 2) containing the coor of nodes of each element.

    Returns:
        int: The index of the element containing the interpolation point.
    """
    num_elements = element_nodes.shape[0]
    for i in range(num_elements):
        if element_nodes[i, 0] <= interpolation_point <= element_nodes[i, 1]:
            return i
    return -1


### C-HiDeNN PART
###Compute adjacency matrix of graph theory
def get_adj_mat(Elem_nodes, nnode, s_patch):
    """ Use graph theory to get nodal connectivity information
    --- Outouts ---
    indices, indptr: sparse matrix pointers
    """
    
    # get adjacency matrix of graph theory based on nodal connectivity
    adj_rows, adj_cols = [], []
    # self connection
    for inode in range(nnode):
        adj_rows += [inode]
        adj_cols += [inode]
    
    for ielem, elem_nodes in enumerate(Elem_nodes):
        for (inode, jnode) in combinations(list(elem_nodes), 2):
            adj_rows += [inode, jnode]
            adj_cols += [jnode, inode]
    adj_values = onp.ones(len(adj_rows), dtype=onp.int32)
    adj_rows = onp.array(adj_rows, dtype=onp.int32)
    adj_cols = onp.array(adj_cols, dtype=onp.int32)
    
    # build sparse matrix
    adj_sp = csc_matrix((adj_values, 
                              (adj_rows, adj_cols)),
                            shape = (nnode, nnode))
    adj_s = csc_matrix((adj_values, 
                              (adj_rows, adj_cols)),
                            shape = (nnode, nnode))
    
    # compute s th power of the adjacency matrix to get s th order of connectivity
    for itr in range(s_patch-1):
        adj_s = adj_s.dot(adj_sp)
    
    indices = adj_s.indices
    indptr = adj_s.indptr
    
    return indices, indptr

def get_adj_mat_list(Elem_nodes_list, s_patch):
    indices_list = []
    indptr_list =  []
    for i in range(len(Elem_nodes_list)):
        nnode = Elem_nodes_list[i].shape[0] + 1
        indices, indptr = get_adj_mat(Elem_nodes_list[i], nnode, s_patch)
        indices_list.append(indices)
        indptr_list.append(indptr)
    return indices_list, indptr_list

#get patch info
def get_dex_max(indices, indptr, s_patch, Elem_nodes): # delete d_c, XY, nnode
    """ Pre-compute the maximum number of elemental patch nodes (edex_max) and nodal patch nodes (ndex_max)
    """    
    dim = 1; nodes_per_elem = 2; nelem = Elem_nodes.shape[0] + 1
    edex_max = (2+2*s_patch)**dim # estimated value of edex_max
    edexes = onp.zeros(nelem, dtype=onp.int32) # (num_elements, )
    ndexes = onp.zeros((nelem, nodes_per_elem), dtype=onp.int32) # static, (nelem, nodes_per_elem)
    
    for ielem, elem_nodes in enumerate(Elem_nodes):
        if len(elem_nodes) == 2 and dim == 1: # 1D Linear element
            nodal_patch_nodes0 = indices[ indptr[elem_nodes[0]] : indptr[elem_nodes[0]+1] ] # global index, # node_idx 0
            nodal_patch_nodes1 = indices[ indptr[elem_nodes[1]] : indptr[elem_nodes[1]+1] ] # global index
            ndexes[ielem, :] = onp.array([len(nodal_patch_nodes0),len(nodal_patch_nodes1)])
            elemental_patch_nodes = onp.unique(onp.concatenate((nodal_patch_nodes0, nodal_patch_nodes1)))  
           
        edexes[ielem] = len(elemental_patch_nodes)
    edex_max = onp.max(edexes)
    ndex_max = onp.max(ndexes)
    return edex_max, ndex_max

def get_patch_info(indices, indptr, edex_max, ndex_max, Elem_nodes): # for block, delete s_patch, d_c, XY
    """ Compute patch information for given elements
    --- Outputs --- lacks explanation here
    """    
    dim = 1; nodes_per_elem = 2; nelem = Elem_nodes.shape[0]
    # Assign memory to variables
    ## Elemental patch
    Elemental_patch_nodes_st = onp.zeros((nelem, edex_max), dtype=onp.int32) # edex_max should be grater than 100!
    edexes = onp.zeros(nelem, dtype=onp.int32) # (num_elements, )
    ## Nodal patch
    Nodal_patch_nodes_st = (-1)*onp.ones((nelem, nodes_per_elem, ndex_max), dtype=onp.int32) # static, (nelem, nodes_per_elem, ndex_max)
    Nodal_patch_nodes_bool = onp.zeros((nelem, nodes_per_elem, ndex_max), dtype=onp.int32) # static, (nelem, nodes_per_elem, ndex_max)
    Nodal_patch_nodes_idx = (-1)*onp.ones((nelem, nodes_per_elem, ndex_max), dtype=onp.int32) # static, (nelem, nodes_per_elem, ndex_max)
    ndexes = onp.zeros((nelem, nodes_per_elem), dtype=onp.int32) # static, (nelem, nodes_per_elem)
    
    
    for ielem, elem_nodes in enumerate(Elem_nodes):
        
        # 1. for loop: nodal_patch_nodes in global nodal index
        for inode_idx, inode in enumerate(elem_nodes):
            nodal_patch_nodes = onp.sort(indices[ indptr[elem_nodes[inode_idx]] : indptr[elem_nodes[inode_idx]+1] ]) # global index
            ndex = len(nodal_patch_nodes)
            ndexes[ielem, inode_idx] = ndex
            Nodal_patch_nodes_st[ielem, inode_idx, :ndex] = nodal_patch_nodes  # global nodal index
            Nodal_patch_nodes_bool[ielem, inode_idx, :ndex] = onp.where(nodal_patch_nodes>=0, 1, 0)
    
        # 2. get elemental_patch_nodes    
        if len(elem_nodes) == 2 and dim == 1: # 1D Linear element
            elemental_patch_nodes = onp.unique(onp.concatenate((Nodal_patch_nodes_st[ielem, 0, :ndexes[ielem, 0]],
                                                                Nodal_patch_nodes_st[ielem, 1, :ndexes[ielem, 1]])))  # node_idx 1
            
        edex = len(elemental_patch_nodes)
        edexes[ielem] = edex
        Elemental_patch_nodes_st[ielem, :edex] = elemental_patch_nodes
        
        # 3. for loop: get nodal_patch_nodes_idx
        for inode_idx, inode in enumerate(elem_nodes):
            nodal_patch_nodes_idx = onp.searchsorted(
                elemental_patch_nodes, Nodal_patch_nodes_st[ielem, inode_idx, :ndexes[ielem, inode_idx]]) # local index
            Nodal_patch_nodes_idx[ielem, inode_idx, :ndexes[ielem, inode_idx]] = nodal_patch_nodes_idx
   
            
    # Convert everything to device array
    Elemental_patch_nodes_st = np.array(Elemental_patch_nodes_st)
    edexes = np.array(edexes)
    Nodal_patch_nodes_st = np.array(Nodal_patch_nodes_st)
    Nodal_patch_nodes_bool = np.array(Nodal_patch_nodes_bool)
    Nodal_patch_nodes_idx = np.array(Nodal_patch_nodes_idx)
    ndexes = np.array(ndexes)
    
    return Elemental_patch_nodes_st, edexes, Nodal_patch_nodes_st, Nodal_patch_nodes_bool, Nodal_patch_nodes_idx, ndexes


def in_range(xi, lb, ub):
    # lb: lower bound, floating number
    # ub: upper bound, floating number
    return np.heaviside(xi-lb,1) * np.heaviside(ub-xi, 0)
@jax.jit
def get_R_cubicSpline(xy, xvi, a_dil):
    zI = np.linalg.norm(xy - xvi)/a_dil
    # R = ((2/3 - 4*zI**2 + 4*zI**3         ) * in_range(zI, 0.0, 0.5) +  \
    #                      (4/3 - 4*zI + 4*zI**2 - 4/3*zI**3) * in_range(zI, 0.5, 1.0))
    def R1(zI):
        R1 = 2/3 - 4*zI**2 + 4*zI**3
        return R1
    def R2(zI): 
        R2 = 4/3 - 4*zI + 4*zI**2 - 4/3*zI**3
        return R2
    def R3(zI): 
        R3 = 0.
        return R3
    R = np.piecewise(zI, [(zI>= 0.) & (zI<= 0.5), (zI> 0.5) & (zI<=1.), zI > 1. ], [R1(zI), R2(zI), R3(zI)])
    #jax.lax.cond((zI >= 0) and (zI <= 1), jax.lax.cond((zI >= 0) and (zI <= 0.5), R1, R2)(zI), R3)(zI)
    return R
v_get_R_cubicSpline = jax.vmap(get_R_cubicSpline, in_axes = (None,0,None))

@jax.jit
def get_R_gaussian1(xy, xvi, a_dil):
    zI = np.linalg.norm(xy - xvi)/a_dil
    R = np.exp(-zI)
    return R
v_get_R_gaussian1 = jax.vmap(get_R_gaussian1, in_axes = (None,0,None))

@jax.jit
def get_R_gaussian2(xy, xvi, a_dil):
    zI = np.linalg.norm(xy - xvi)/a_dil
    R = np.exp(-zI**2)
    return R
v_get_R_gaussian2 = jax.vmap(get_R_gaussian2, in_axes = (None,0,None))

@jax.jit
def get_R_gaussian3(xy, xvi, a_dil):
    zI = np.linalg.norm(xy - xvi)/a_dil
    R = np.exp(-zI**3)
    return R
v_get_R_gaussian3 = jax.vmap(get_R_gaussian3, in_axes = (None,0,None))

@jax.jit
def get_R_gaussian4(xy, xvi, a_dil):
    zI = np.linalg.norm(xy - xvi)/a_dil
    R = np.exp(-zI**4)
    return R
v_get_R_gaussian4 = jax.vmap(get_R_gaussian4, in_axes = (None,0,None))

@jax.jit
def get_R_gaussian5(xy, xvi, a_dil):
    zI = np.linalg.norm(xy - xvi)/a_dil
    R = np.exp(-zI**5)
    return R
v_get_R_gaussian5 = jax.vmap(get_R_gaussian5, in_axes = (None,0,None))

@partial(jax.jit, static_argnames=['ndex_max', 'mbasis', 'radial_basis', 'dim']) # This will slower the function
def Compute_RadialBasis_1D(xy, xv, ndex, ndex_max, nodal_patch_nodes_bool, 
                         a_dil, mbasis, radial_basis, dim):
    """ 
    ******THIS ONLY WORKS FOR 1D DIM (WHICH MEANS TD)*******
    --- Inputs ---
    # xy: point of interest (dim,)
    # xv: ndoal coordinates of patch nodes. (num_nodes)
    # ndex: number of nodse in the nodal patch
    # ndex_max: max of ndex, precomputed value
    # nodal_patch_nodes_bool: boolean vector that tells ~~~
    # a_dil: dilation parameter for cubic spline
    # mbasis: number of polynomial terms
    
    """
    
    RP = np.zeros(ndex_max + mbasis, dtype=np.double)
    
    if radial_basis == 'cubicSpline':

        RP = RP.at[:ndex_max].set(v_get_R_cubicSpline(xy, xv, a_dil) * nodal_patch_nodes_bool)
    if radial_basis == 'gaussian1':
        RP = RP.at[:ndex_max].set(v_get_R_gaussian1(xy, xv, a_dil) * nodal_patch_nodes_bool)        
    if radial_basis == 'gaussian2':
        RP = RP.at[:ndex_max].set(v_get_R_gaussian2(xy, xv, a_dil) * nodal_patch_nodes_bool)        
    if radial_basis == 'gaussian3':
        RP = RP.at[:ndex_max].set(v_get_R_gaussian3(xy, xv, a_dil) * nodal_patch_nodes_bool)        
    if radial_basis == 'gaussian5':
        RP = RP.at[:ndex_max].set(v_get_R_gaussian3(xy, xv, a_dil) * nodal_patch_nodes_bool)        

    
    if mbasis > 0: # 1st
        RP = RP.at[ndex_max   : ndex_max+ 2].set(np.array([1 , xy[0] ]))   # N 1, x
        
    if mbasis > 2: # 2nd
        RP = RP.at[ndex_max+ 2: ndex_max+ 3].set(np.array([xy[0]**2]))   # N x^2
        
    if mbasis > 3: # 3rd
        RP = RP.at[ndex_max+ 3: ndex_max+ 4].set(np.array([xy[0]**3]))   # N x^3
        
    if mbasis > 4: # 4th
        RP = RP.at[ndex_max+ 4: ndex_max+ 5].set(np.array([xy[0]**4]))   # N x^4
        
    return RP    

v_Compute_RadialBasis_1D = jax.vmap(Compute_RadialBasis_1D, in_axes = (0,None,None,None,None,
                                                                                   None,None,None,None), out_axes=1)
Compute_RadialBssis_der = jax.jacfwd(Compute_RadialBasis_1D, argnums=0)


@partial(jax.jit, static_argnames=['ndex_max','a_dil','mbasis','radial_basis','dim']) # unneccessary
def get_G(ndex, nodal_patch_nodes, nodal_patch_nodes_bool, XY, ndex_max, a_dil, mbasis, radial_basis, dim):
    # nodal_patch_nodes_bool: (ndex_max,)
    G = np.zeros((ndex_max + mbasis, ndex_max + mbasis), dtype=np.double)
    xv = XY[nodal_patch_nodes,:]
    RPs = v_Compute_RadialBasis_1D(xv, xv, ndex, ndex_max, nodal_patch_nodes_bool, 
                                  a_dil, mbasis, radial_basis, dim) # (ndex_max + mbasis, ndex_max)
    
    G = G.at[:,:ndex_max].set(RPs * nodal_patch_nodes_bool[None,:])                        
    
    # Make symmetric matrix
    G = np.tril(G) + np.triu(G.T, 1)
    
    # Build diagonal terms to nullify dimensions
    Imat = np.eye(ndex_max) * np.abs(nodal_patch_nodes_bool-1)[:,None]
    G = G.at[:ndex_max,:ndex_max].add(Imat)
    return G # G matrix

vv_get_G = jax.vmap(jax.vmap(get_G, in_axes = (0,0,0,None,None,None,None,None,None)), in_axes = (0,0,0,None,None,  None,None,None,None))
#Gs: (num_cells, num_nodes, ndex_max+mbasis, ndex_max+mbasis)

# Assuming phi_org is your matrix
def is_singular(matrix):
    # Check if matrix is square first
    if matrix.shape[0] != matrix.shape[1]:
        return True  # Non-square matrices are not invertible
    
    # Compute determinant
    det = np.linalg.det(matrix)
    
    # Consider numerical precision - a very small determinant means near-singular
    return np.abs(det) < 1e-10


@partial(jax.jit, static_argnames=['ndex_max','edex_max','a_dil','mbasis','radial_basis','dim']) # must
def get_Phi(G, nodal_patch_nodes, nodal_patch_nodes_bool, nodal_patch_nodes_idx, ndex, shape_val, elem_nodes,
            XY, ndex_max, edex_max, a_dil, mbasis, radial_basis, dim): # 15
    
    # shape_val (quad_num, nodes_per_elem) 
    xy_elem = XY[elem_nodes,:] # (nodes_per_elem, dim)
    xv = XY[nodal_patch_nodes,:] # (nodes_per_elem, dim)
    
    xy = np.sum(shape_val[:, None] * xy_elem, axis=0, keepdims=False)
    RP_1D = Compute_RadialBasis_1D(xy, xv, ndex, ndex_max, nodal_patch_nodes_bool, 
                              a_dil, mbasis, radial_basis, dim) # (edex_max,) but only 'ndex+1' nonzero terms
    
    #derivative of RP
    RP_der = Compute_RadialBssis_der(xy, xv, ndex, ndex_max, nodal_patch_nodes_bool, 
                              a_dil, mbasis, radial_basis, dim) 
    # at nodals, autodiff give me infinity numbers, it should be zero, i switch it here
    RP_der = np.nan_to_num(RP_der)
    RP = np.column_stack((RP_1D, RP_der))

    ## Standard matrix solver
    phi_org = np.linalg.solve(G.T, RP)[:ndex_max,:] * nodal_patch_nodes_bool[:, None]
    # is_sing = is_singular(G.T)
    
    ## Assemble
    phi = np.zeros((edex_max + 1, 1+dim))  # trick, add dummy node at the end
    phi = phi.at[nodal_patch_nodes_idx, :].set(phi_org) 
    phi = phi[:edex_max,:] # trick, delete dummy node
    
    return phi #edex_max, 1+dim
#vvv_get_Phi: [nelem_per, quad_num, nodes_per_elem, edex_max, 1+dim]
vvv_get_Phi = jax.vmap(jax.vmap(jax.vmap(get_Phi, in_axes = (0,0,0,0,0,None,None,None,None,None,None,None,None,None)), #vmap for nodes_per_elem
                                in_axes = (None,None,None,None,None,0,None,None,None,None,None,None,None,None)), #vmap for quad_num # shape_val (quad_num, nodes_per_elem) 
                                in_axes = (0,0,0,0,0,None,0,None,None,None,None,None,None,None))  #vmap for nelem -> nelem for fine mesh

#get N_tilda
def get_CFEM_shape_fun(elem_idx, nelem_per,
                       XY, Elem_nodes, shape_vals, Gauss_Num, dim, elem_type, nodes_per_elem,
                       indices, indptr, s_patch, edex_max, ndex_max,
                        a_dil, mbasis, radial_basis):
    """_summary_
    Args:
        elem_idx (_type_): _description_
        nelem_per (_type_): _description_
        XY (_type_): _description_
        Elem_nodes (_type_): _description_
        shape_vals (_type_): _description_
        Gauss_Num (_type_): _description_
        dim (_type_): _description_
        elem_type (_type_): _description_
        nodes_per_elem (_type_): _description_
        indices (_type_): _description_
        indptr (_type_): _description_
        s_patch (_type_): _description_
        edex_max (_type_): _description_
        ndex_max (_type_): _description_
        a_dil (_type_): _description_
        mbasis (_type_): _description_
        radial_basis (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Define parameters
    quad_num = Gauss_Num**dim
    Elem_nodes = Elem_nodes[elem_idx] # gpu
    shape_grads_physical, JxW = get_shape_grads(Gauss_Num, dim, elem_type, XY, Elem_nodes) # gpu

    # Get patch information
    (Elemental_patch_nodes_st, edexes,
     Nodal_patch_nodes_st, Nodal_patch_nodes_bool, Nodal_patch_nodes_idx, ndexes
     ) = get_patch_info(indices, indptr, edex_max, ndex_max, Elem_nodes)                         
    # Compute assembled moment matrix G
    #Gs = vv_get_G(ndexes, Nodal_patch_nodes_st, Nodal_patch_nodes_bool, XY, ndex_max, a_dil, mbasis)
    Gs = vv_get_G(ndexes, Nodal_patch_nodes_st, Nodal_patch_nodes_bool, XY, ndex_max, a_dil, mbasis, radial_basis, dim)
    
    Phi = vvv_get_Phi(Gs, Nodal_patch_nodes_st, Nodal_patch_nodes_bool, Nodal_patch_nodes_idx, ndexes, shape_vals, Elem_nodes,
                XY, ndex_max, edex_max, a_dil, mbasis, radial_basis, dim) # [nelem_per_block, quad_num, nodes_per_elem, edex_max, 1+dim]
    if np.any(np.isnan(Phi)):  # Works outside JIT
        print("Phi is NaN")
    #shape_vals: (num_quad, num_node)
    N_til = np.sum(shape_vals[None, :, :, None]*Phi[:,:,:,:,0], axis=2) # (nelem, quad_num, edex_max)
    #shape_grads_physical: (num_cell, num_quad, num_node, dim)
    Grad_N_til = (np.sum(shape_grads_physical[:, :, :, None, :]*Phi[:,:,:,:,:1], axis=2)  # (nelem, quad_num, edex_max, dim = 1)
                      + np.sum(shape_vals[None, :, :, None, None]*Phi[:,:,:,:,1:], axis=2) )
    
    # Check partition of unity
    if not ( np.allclose(np.sum(N_til, axis=2), np.ones((nelem_per, quad_num), dtype=np.double)) and
            np.allclose(np.sum(Grad_N_til, axis=2), np.zeros((nelem_per, quad_num, dim), dtype=np.double)) ):
        print(f"PoU Check failed at element {elem_idx[0]}~{elem_idx[-1]}")
        PoU_Check_N = (np.linalg.norm(np.sum(N_til, axis=2) - np.ones((nelem_per, quad_num), dtype=np.float64))**2/(nelem_per*quad_num))**0.5
        PoU_Check_Grad_N = (np.linalg.norm(np.sum(Grad_N_til, axis=2))**2/(nelem_per*quad_num*dim))**0.5
        print(f'PoU check N / Grad_N: {PoU_Check_N:.4e} / {PoU_Check_Grad_N:.4e}')
        
    return N_til, Grad_N_til, JxW, Elemental_patch_nodes_st


def get_Gs_inv(x_grid, Elem_nodes,
                       indices, indptr, edex_max, ndex_max,
                        a_dil, mbasis):
    """_summary_
    Args:
        x_grid (num_nodes, 1): _description_
        Elem_nodes (_type_): _description_
        shape_vals (_type_): _description_
        Gauss_Num (_type_): _description_
        dim (_type_): _description_
        elem_type (_type_): _description_
        nodes_per_elem (_type_): _description_
        indices (_type_): _description_
        indptr (_type_): _description_
        s_patch (_type_): _description_
        edex_max (_type_): _description_
        ndex_max (_type_): _description_
        a_dil (_type_): _description_
        mbasis (_type_): _description_
        radial_basis (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Define parameters
    dim = 1
    radial_basis = 'cubicSpline'
    elem_type = 'D1LN2N'
    # Get patch information
    (_, _,
     Nodal_patch_nodes_st, Nodal_patch_nodes_bool, Nodal_patch_nodes_idx, ndexes
     ) = get_patch_info(indices, indptr, edex_max, ndex_max, Elem_nodes)                         
    # Compute assembled moment matrix G
    #Gs = vv_get_G(ndexes, Nodal_patch_nodes_st, Nodal_patch_nodes_bool, XY, ndex_max, a_dil, mbasis)
    Gs = vv_get_G(ndexes, Nodal_patch_nodes_st, Nodal_patch_nodes_bool, x_grid, ndex_max, a_dil, mbasis, radial_basis, dim)
    
    Gs_inv = np.linalg.inv(Gs)
    
    shape_grads_physical, _ = get_shape_grads(0, dim, elem_type, x_grid, Elem_nodes) # TODO: only for linear element
    return {
        'Gs_inv': Gs_inv,
        'Nodal_patch_nodes_st': Nodal_patch_nodes_st,
        'Nodal_patch_nodes_bool': Nodal_patch_nodes_bool,
        'ndexes': ndexes,
        'ndex_max': ndex_max,
        'a_dil': a_dil,
        'mbasis': mbasis,
        'shape_grads_physical': shape_grads_physical
    }

def find_ielem(xi, grid):
    i = np.searchsorted(grid, xi) - 1
    i = np.where(i < 0, 0, i) # i: element index
    norm_distance = (xi - grid[i]) / (grid[i + 1] - grid[i])
    return i, norm_distance


                         
def get_phi_anypt(xi, ielem, inode, values, grid, Gs_inv, nodal_patch_nodes_st, nodal_patch_nodes_bool, ndexes, ndex_max, a_dil, mbasis):
    # xi: (1,)
    # ielem: element index [1]
    # inode: node index (1)
    # return: result: values at xi;  result_der: derivative of values at xi
    Ginv = Gs_inv[ielem[0], inode]  # (K+P+1, K+P+1) where K: ndex_max
    nodal_patch_nodes = nodal_patch_nodes_st[ielem,inode]
    RP = Compute_RadialBasis_1D(xi.reshape(-1), grid[nodal_patch_nodes].reshape(-1, 1), 
                                     ndexes[ielem,inode].reshape(-1), ndex_max, nodal_patch_nodes_bool[ielem,inode].reshape(-1), a_dil, mbasis, 'cubicSpline', 1) # (K+P+1)
    # derivative of RP
    RP_der = Compute_RadialBssis_der(xi.reshape(-1), grid[nodal_patch_nodes].reshape(-1, 1), 
                                     ndexes[ielem,inode].reshape(-1), ndex_max, nodal_patch_nodes_bool[ielem,inode].reshape(-1), a_dil, mbasis, 'cubicSpline', 1).reshape(-1) # (K+P+1) 

    # # at nodals, autodiff give me infinity numbers, it should be zero, i switch it here
    RP_der = np.nan_to_num(RP_der)   
    RP_Ginv = np.tensordot(RP, Ginv, axes=(0,1))[:ndex_max]
    RP_der_Ginv = np.tensordot(RP_der, Ginv, axes=(0,1))[:ndex_max]
    result = np.sum(RP_Ginv * np.take(values, nodal_patch_nodes), keepdims=False)
    result_der = np.sum(RP_der_Ginv * np.take(values, nodal_patch_nodes), keepdims=False)
    return result, result_der

def CFEM_anypt(xi, grid, values, cfem_mesh_dict):
    """
    xi: (1,) point in the domain
    values (num_points, 1)
    input grid (num_grid)
    Gs_inv: (-, K+P+1, K+P+1) inverse of the moment matrix
    --- values: (J,) nodal values
    """
  
    ielem, norm_distance = find_ielem(xi, grid)
    
    Gs_inv = cfem_mesh_dict['Gs_inv']
    nodal_patch_nodes_st = cfem_mesh_dict['Nodal_patch_nodes_st']
    nodal_patch_nodes_bool = cfem_mesh_dict['Nodal_patch_nodes_bool']
    ndexes = cfem_mesh_dict['ndexes']
    ndex_max = cfem_mesh_dict['ndex_max']
    a_dil = cfem_mesh_dict['a_dil']
    mbasis = cfem_mesh_dict['mbasis']
    
    # linear finite * W fun
    qerry_value = get_phi_anypt(xi, ielem, 0, values, grid, Gs_inv, nodal_patch_nodes_st, nodal_patch_nodes_bool, ndexes, ndex_max, a_dil, mbasis)[0] \
        * (1-norm_distance) + get_phi_anypt(xi, ielem, 1, values, grid, Gs_inv, nodal_patch_nodes_st, nodal_patch_nodes_bool, ndexes, ndex_max, a_dil, mbasis)[0] * (norm_distance)
    return qerry_value


def CFEM_der_anypt(xi, grid, values, cfem_mesh_dict):
    """
    obtain the derivative of the qerried point
    xi: (1,) point in the domain
    values (num_points, 1)
    input grid (num_grid)
    Gs_inv: (-, K+P+1, K+P+1) inverse of the moment matrix
    --- values: (J,) nodal values
    """
  
    ielem, norm_distance = find_ielem(xi, grid)
    
    Gs_inv = cfem_mesh_dict['Gs_inv']
    nodal_patch_nodes_st = cfem_mesh_dict['Nodal_patch_nodes_st']
    nodal_patch_nodes_bool = cfem_mesh_dict['Nodal_patch_nodes_bool']
    ndexes = cfem_mesh_dict['ndexes']
    ndex_max = cfem_mesh_dict['ndex_max']
    a_dil = cfem_mesh_dict['a_dil']
    mbasis = cfem_mesh_dict['mbasis']
    shape_grads_physical = cfem_mesh_dict['shape_grads_physical'] #(nelem, quad_num, nodes_per_elem, dim)
    # derivative of linear finite * W fun = linear finite' * W fun + linear finite * W fun'
    shape_grads = shape_grads_physical[ielem, 0, :, 0].reshape(-1) # (nodes_per_elem) # TODO: only applies to linear elements
    part1 = shape_grads[0] * get_phi_anypt(xi, ielem, 0, values, grid, Gs_inv, nodal_patch_nodes_st, nodal_patch_nodes_bool, ndexes, ndex_max, a_dil, mbasis)[0] \
        + shape_grads[1] * get_phi_anypt(xi, ielem, 1, values, grid, Gs_inv, nodal_patch_nodes_st, nodal_patch_nodes_bool, ndexes, ndex_max, a_dil, mbasis)[0]
    part2 = get_phi_anypt(xi, ielem, 0, values, grid, Gs_inv, nodal_patch_nodes_st, nodal_patch_nodes_bool, ndexes, ndex_max, a_dil, mbasis)[1] \
        * (1-norm_distance) + get_phi_anypt(xi, ielem, 1, values, grid, Gs_inv, nodal_patch_nodes_st, nodal_patch_nodes_bool, ndexes, ndex_max, a_dil, mbasis)[1] * (norm_distance)
    qerry_derivative = part1 + part2
    return qerry_derivative

#input xi (num_points, 1)
#input values (num_points, 1)
#input grid (num_grid)
#return shape (num_points, 1) ->needs to be reshaped properly when used
vmap_CFEM_anypt = jax.vmap(CFEM_anypt, in_axes=(0, None, None, None))
vmap_CFEM_der_anypt = jax.vmap(CFEM_der_anypt, in_axes=(0, None, None, None))