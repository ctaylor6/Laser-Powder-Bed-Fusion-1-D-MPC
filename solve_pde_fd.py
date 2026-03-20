import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

def solve_pde_fd(f, x, t, initial_condition=None):
    """
    Solve the PDE ∂u/∂t - ∂²u/∂x² = f(x,t) with Crank-Nicolson.
    
    Args:
        f (callable or array): [num_pt_x, num_pt_t]
        t_max (float): Maximum time
        nx (int): Spatial grid points
        nt (int): Time steps
        initial_condition (array or callable): Initial condition u(x,0). If None, defaults to zero.
    
    Returns:
        tuple: (x_grid, t_grid, solution)
    """
    # Grid setup
   
    nx = len(x)
    nt = len(t)
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    r = dt / (2 * dx**2)  # Crank-Nicolson parameter

    # Initialize solution
    u = np.zeros((nt, nx))

    # Set initial condition
    if initial_condition is not None:
        if callable(initial_condition):
            u[0, :] = initial_condition(x)
        else:
            u[0, :] = initial_condition  # Assume it's an array of shape (nx,)
    # Construct sparse matrix (implicit part)
    main_diag = np.ones(nx-2) * (1 + 2*r)
    off_diag = np.ones(nx-3) * (-r)
    A = diags([off_diag, main_diag, off_diag], [-1, 0, 1], format='csr')

    # Time-stepping
    for n in range(nt-1):
        t_n = t[n]
        t_n1 = t[n+1]
        
        # Evaluate f(x,t) at current and next time steps
        if callable(f):
            f_current = f(x[1:-1], t_n)
            f_next = f(x[1:-1], t_n1)
        else:  # Assume it's an array of shape (nx, nt)
            f_current = f[1:-1, n]
            f_next = f[1:-1, n+1]
        
        # Explicit RHS vector
        b = u[n, 1:-1] + r * (u[n, :-2] - 2*u[n, 1:-1] + u[n, 2:]) + dt * (f_current + f_next)/2
        
        # Solve the system
        u[n+1, 1:-1] = spsolve(A, b)
        
        # Enforce boundary conditions (Dirichlet: u=0 at boundaries)
        u[n+1, 0] = 0
        u[n+1, -1] = 0
    
    return x, t, u