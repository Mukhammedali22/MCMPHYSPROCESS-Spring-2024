import numpy as np
from numpy import (pi, exp, sin, cos)
from numba import jit


# Returns numpy ndarray
# Passing by reference
def Jacobi_method(P:np.ndarray, U:np.ndarray, V:np.ndarray, 
                  N, M, dx, dy, dt, rho) -> np.ndarray:
    """Jacobi method for solving 2D Poisson equation"""
    P_old = P
    U_new = U
    V_new = V

    P_new = np.zeros_like(P)

    P_new[1:M-1, 1:N-1] = (dy**2*(P_old[1:M-1, 2:N] + P_old[1:M-1, 0:N-2]) \
        + dx**2*(P_old[2:M, 1:N-1] + P_old[0:M-2, 1:N-1])) \
            / (2*(dx**2 + dy**2)) \
        - dx**2*dy**2*rho \
            / (2*dt*(dx**2 + dy**2)) \
        * ((U_new[1:M-1, 2:N] - U_new[1:M-1, 0:N-2]) \
            / (2*dx) \
        + (V_new[2:M, 1:N-1] - V_new[0:M-2, 1:N-1]) \
            / (2*dy))
    
    return P_new


# Returns numpy ndarray
# Passing by reference
@jit(nopython=True)
def Numba_Jacobi_method(P:np.ndarray, U:np.ndarray, V:np.ndarray, 
                        N, M, dx, dy, dt, rho) -> np.ndarray:
    """Boosted Jacobi method for solving 2D Poisson equation"""
    P_old = P
    U_new = U
    V_new = V

    P_new = np.zeros_like(P)

    P_new[1:M-1, 1:N-1] = (dy**2*(P_old[1:M-1, 2:N] + P_old[1:M-1, 0:N-2]) \
        + dx**2*(P_old[2:M, 1:N-1] + P_old[0:M-2, 1:N-1])) \
            / (2*(dx**2 + dy**2)) \
        - dx**2*dy**2*rho \
            / (2*dt*(dx**2 + dy**2)) \
        * ((U_new[1:M-1, 2:N] - U_new[1:M-1, 0:N-2]) \
            / (2*dx) \
        + (V_new[2:M, 1:N-1] - V_new[0:M-2, 1:N-1]) \
            / (2*dy))
    
    return P_new


# Returns numpy ndarray
# Passing by reference
@jit(nopython=True)
def Numba_Gauss_Seidel_method(P1:np.ndarray, P0:np.ndarray, U:np.ndarray, 
                       V:np.ndarray, N, M, dx, dy, dt, rho) -> np.ndarray:
    """Boosted Gauss-Seidel method for solving 2D Poisson equation"""
    P_old = P0
    P_new = P1.copy()
    U_new = U
    V_new = V

    for j in range(1, M-1):
        for i in range(1, N-1):
            P_new[j, i] = (dy**2*(P_old[j, i+1] + P_new[j, i-1]) \
                + dx**2*(P_old[j+1, i] + P_new[j-1, i])) \
                    / (2*(dx**2 + dy**2)) \
                - dx**2*dy**2*rho \
                    / (2*dt*(dx**2 + dy**2)) \
                * ((U_new[j, i+1] - U_new[j, i-1]) \
                    / (2*dx) \
                + (V_new[j+1, i] - V_new[j-1, i]) \
                    / (2*dy))

    return P_new


# Returns numpy ndarray
# Passing by reference
@jit(nopython=True)
def Numba_Relaxation_method(P1:np.ndarray, P0:np.ndarray, U:np.ndarray, 
                             V:np.ndarray, N, M, dx, dy, dt, rho, w=1.5) -> np.ndarray:
    """Boosted Over relaxation method for solving 2D Poisson equation"""
    P_old = P0
    P_new = P1.copy()
    U_new = U
    V_new = V
    
    for j in range(1, M-1):
        for i in range(1, N-1):
            P_new[j, i] = w*(dy**2*(P_old[j, i+1] + P_new[j, i-1]) \
                + dx**2*(P_old[j+1, i] + P_new[j-1, i])) \
                    / (2*(dx**2 + dy**2)) \
                - w*dx**2*dy**2*rho \
                    / (2*dt*(dx**2 + dy**2)) \
                * ((U_new[j, i+1] - U_new[j, i-1]) \
                    / (2*dx) \
                + (V_new[j+1, i] - V_new[j-1, i]) \
                    / (2*dy)) \
                + (1 - w)*P_old[j][i]
    
    return P_new
