import numpy as np
from numpy import (pi, exp, sin, cos)
from numba import jit


# Returns numpy ndarray
# Passing by reference
def Explicit_method(S:np.ndarray, U:np.ndarray, V:np.ndarray, N, M, dx, dy, dt, nu) -> np.ndarray:
    # Reference
    U_old = U
    V_old = V
    S_old = S
    
    S_new = np.zeros_like(S)
    
    S_new[1:M-1, 1:N-1] = S_old[1:M-1, 1:N-1] + dt*(
        - U_old[1:M-1, 1:N-1]*(S_old[1:M-1, 2:N] - S_old[1:M-1, 0:N-2]) \
                                / (2*dx) \
        - V_old[1:M-1, 1:N-1]*(S_old[2:M, 1:N-1] - S_old[0:M-2, 1:N-1]) \
                                / (2*dy) \
        + nu*(
            (S_old[1:M-1, 2:N] - 2*S_old[1:M-1, 1:N-1] + S_old[1:M-1, 0:N-2]) \
                / dx**2 \
            + (S_old[2:M, 1:N-1] - 2*S_old[1:M-1, 1:N-1] + S_old[0:M-2, 1:N-1]) \
                / dy**2))
    
    return S_new


# Returns numpy ndarray
# Passing by reference
def FSM(S:np.ndarray, U:np.ndarray, V:np.ndarray, N, M, dx, dy, dt, nu) -> np.ndarray:
    U_old = U
    V_old = V
    S_old = S
    
    S_new = np.zeros_like(S)
    
    N1 = int(0.2 * N)
    N2 = int(0.3 * N)
    N3 = int(0.7 * N)
    N4 = int(0.8 * N)

    A = np.zeros((M, N))
    B = np.zeros((M, N))
    C = np.zeros((M, N))
    D = np.zeros((M, N))

    alpha = np.zeros((M, N))
    beta = np.zeros((M, N))
    
    # Finding U^(n+1/2)
    A[0:M, 0:N] = -nu / (2*dx**2) + U_old[0:M, 0:N] / (2*dx)
    B[0:M, 0:N] = 1 / dt + nu / (dx**2) - U_old[0:M, 0:N] / (2*dx)
    C[0:M, 0:N] = -nu / (2*dx**2)
    
    D[1:M-1, 1:N-1] = S_old[1:M-1, 1:N-1] / dt + 0.5*(
        nu*(S_old[1:M-1, 2:N] - 2*S_old[1:M-1, 1:N-1] + S_old[1:M-1, 0:N-2]) \
            / (dx**2) \
        - U_old[1:M-1, 1:N-1] * (S_old[1:M-1, 2:N] - S_old[1:M-1, 1:N-1]) \
            / dx) \
        + nu*(S_old[2:M, 1:N-1] - 2*S_old[1:M-1, 1:N-1] + S_old[0:M-2, 1:N-1]) \
            / (dy**2) \
        - V_old[1:M-1, 1:N-1] * (S_old[2:M, 1:N-1] - S_old[1:M-1, 1:N-1]) \
            / dy
    
    # Thomas algorithm for x
    # U(t, x=0, y) = 0
    alpha[0:M, 1] = 0
    beta[0:M, 1] = 0
        
    for i in range(1, N-1):
        alpha[1:M-1, i+1] = -A[1:N-1, i] \
            / (B[1:M-1, i] + C[1:N-1, i]*alpha[1:N-1, i])
        beta[1:M-1, i+1] = (D[1:M-1, i] - C[1:N-1, i]*beta[1:M-1, i]) \
            / (B[1:M-1, i] + C[1:N-1, i]*alpha[1:N-1, i])
        
    # U^(n+1/2)
    S_new[0:M, N-1] = S_old[0:M, N-1]
    for i in range(N-2, -1, -1):
        S_new[1:M-1, i] = alpha[1:M-1, i+1]*S_new[1:M-1, i+1] \
                            + beta[1:M-1, i+1]   
    
    # Finding U^(n+1)
    A[0:M, 0:N] = -nu / (2*dy**2) + V_old[0:M, 0:N] / (2*dy)
    B[0:M, 0:N] = 1 / dt + nu / (dy**2) - V_old[0:M, 0:N] / (2*dy)
    C[0:M, 0:N] = -nu / (2*dy**2)

    # U_new is U^(n+1/2) now
    D[1:M-1, 1:N-1] = S_new[1:M-1, 1:N-1] / dt - 0.5*(
        nu*(S_old[2:M, 1:N-1] - 2*S_old[1:M-1, 1:N-1] + S_old[0:M-2, 1:N-1]) \
            / (dy**2) \
        - V_old[1:M-1, 1:N-1] * (S_old[2:M, 1:N-1] - S_old[1:M-1, 1:N-1]) \
            / dy)

    # Thomas algorithm for y
    # Uy(t, 0<x<0.2, y=0) = 0
    alpha[1, 0:N1] = 1
    beta[1, 0:N1] = 0
    # U(t, 0.2<x<0.8, y=0) = 0
    alpha[1, N1:N4] = 0
    beta[1, N1:N4] = 0
    # Uy(t, 0.8<x<1, y=0) = 0
    alpha[1, N4:N] = 1
    beta[1, N4:N] = 0

    for j in range(1, M-1):
        alpha[j+1, 1:N-1] = -A[j, 1:N-1] \
            / (B[j, 1:N-1] + C[j, 1:N-1]*alpha[j, 1:N-1])
        beta[j+1, 1:N-1] = (D[j, 1:N-1] - C[j, 1:N-1]*beta[j, 1:N-1]) \
            / (B[j, 1:N-1] + C[j, 1:N-1]*alpha[j, 1:N-1])
        
    # U^(n+1)
    S_new[M-1, 0:N] = S_old[M-1, 0:N]
    for j in range(M-2, -1, -1):
        S_new[j, 1:N-1] = alpha[j+1, 1:N-1]*S_new[j+1, 1:N-1] \
                            + beta[j+1, 1:N-1]
    
    return S_new


# Returns numpy ndarray
# Passing by reference
def ADM(S:np.ndarray, U:np.ndarray, V:np.ndarray, N, M, dx, dy, dt, nu) -> np.ndarray:
    U_old = U
    V_old = V
    S_old = S
    
    S_new = np.zeros_like(S)
    
    N1 = int(0.2 * N)
    N2 = int(0.3 * N)
    N3 = int(0.7 * N)
    N4 = int(0.8 * N)

    A = np.zeros((M, N))
    B = np.zeros((M, N))
    C = np.zeros((M, N))
    D = np.zeros((M, N))

    alpha = np.zeros((M, N))
    beta = np.zeros((M, N))
    
    # Finding U^(n+1/2)
    A[0:M, 0:N] = -nu / (dx**2) + U_old[0:M, 0:N] / (dx)
    B[0:M, 0:N] = 1 / dt + 2*nu / (dx**2) - U_old[0:M, 0:N] / (dx)
    C[0:M, 0:N] = -nu / (dx**2)
    
    D[1:M-1, 1:N-1] = S_old[1:M-1, 1:N-1] / dt \
        + nu*(S_old[2:M, 1:N-1] - 2*S_old[1:M-1, 1:N-1] + S_old[0:M-2, 1:N-1]) \
            / (dy**2) \
        - V_old[1:M-1, 1:N-1] * (S_old[2:M, 1:N-1] - S_old[1:M-1, 1:N-1]) \
            / dy
    
    # Thomas algorithm for x
    # U(t, x=0, y) = 0
    alpha[0:M, 1] = 0
    beta[0:M, 1] = 0
        
    for i in range(1, N-1):
        alpha[1:M-1, i+1] = -A[1:N-1, i] \
            / (B[1:M-1, i] + C[1:N-1, i]*alpha[1:N-1, i])
        beta[1:M-1, i+1] = (D[1:M-1, i] - C[1:N-1, i]*beta[1:M-1, i]) \
            / (B[1:M-1, i] + C[1:N-1, i]*alpha[1:N-1, i])
        
    # U^(n+1/2)
    S_new[0:M, N-1] = S_old[0:M, N-1]
    for i in range(N-2, -1, -1):
        S_new[1:M-1, i] = alpha[1:M-1, i+1]*S_new[1:M-1, i+1] \
                            + beta[1:M-1, i+1]   
    
    # Finding U^(n+1)
    A[0:M, 0:N] = -nu / (dy**2) + V_old[0:M, 0:N] / (dy)
    B[0:M, 0:N] = 1 / dt + 2*nu / (dy**2) - V_old[0:M, 0:N] / (dy)
    C[0:M, 0:N] = -nu / (dy**2)

    # U_new is U^(n+1/2) now
    D[1:M-1, 1:N-1] = S_new[1:M-1, 1:N-1] / dt \
        + nu*(S_new[1:M-1, 2:N] - 2*S_new[1:M-1, 1:N-1] + S_new[1:M-1, 0:N-2]) \
            / (dx**2) \
        - U_old[1:M-1, 1:N-1] * (S_new[1:M-1, 2:N] - S_new[1:M-1, 1:N-1]) \
            / dx

    # Thomas algorithm for y
    # Uy(t, 0<x<0.2, y=0) = 0
    alpha[1, 0:N1] = 1
    beta[1, 0:N1] = 0
    # U(t, 0.2<x<0.8, y=0) = 0
    alpha[1, N1:N4] = 0
    beta[1, N1:N4] = 0
    # Uy(t, 0.8<x<1, y=0) = 0
    alpha[1, N4:N] = 1
    beta[1, N4:N] = 0
    
    for j in range(1, M-1):
        alpha[j+1, 1:N-1] = -A[j, 1:N-1] \
            / (B[j, 1:N-1] + C[j, 1:N-1]*alpha[j, 1:N-1])
        beta[j+1, 1:N-1] = (D[j, 1:N-1] - C[j, 1:N-1]*beta[j, 1:N-1]) \
            / (B[j, 1:N-1] + C[j, 1:N-1]*alpha[j, 1:N-1])
        
    # U^(n+1)
    S_new[M-1, 0:N] = S_old[M-1, 0:N]
    for j in range(M-2, -1, -1):
        S_new[j, 1:N-1] = alpha[j+1, 1:N-1]*S_new[j+1, 1:N-1] \
                            + beta[j+1, 1:N-1]
                            
    return S_new


# Returns numpy ndarray
# Passing by reference
@jit(nopython=True)
def Numba_Explicit_method(S:np.ndarray, U:np.ndarray, V:np.ndarray, N, M, dx, dy, dt, nu) -> np.ndarray:
    # Reference
    U_old = U
    V_old = V
    S_old = S
    
    S_new = np.zeros_like(S)
    
    S_new[1:M-1, 1:N-1] = S_old[1:M-1, 1:N-1] + dt*(
        - U_old[1:M-1, 1:N-1]*(S_old[1:M-1, 2:N] - S_old[1:M-1, 0:N-2]) \
                                / (2*dx) \
        - V_old[1:M-1, 1:N-1]*(S_old[2:M, 1:N-1] - S_old[0:M-2, 1:N-1]) \
                                / (2*dy) \
        + nu*(
            (S_old[1:M-1, 2:N] - 2*S_old[1:M-1, 1:N-1] + S_old[1:M-1, 0:N-2]) \
                / dx**2 \
            + (S_old[2:M, 1:N-1] - 2*S_old[1:M-1, 1:N-1] + S_old[0:M-2, 1:N-1]) \
                / dy**2))
    
    return S_new


# Returns numpy ndarray
# Passing by reference
@jit(nopython=True)
def Numba_FSM(S:np.ndarray, U:np.ndarray, V:np.ndarray, N, M, dx, dy, dt, nu) -> np.ndarray:
    U_old = U
    V_old = V
    S_old = S
    
    S_new = np.zeros_like(S)
    
    N1 = int(0.2 * N)
    N2 = int(0.3 * N)
    N3 = int(0.7 * N)
    N4 = int(0.8 * N)

    A = np.zeros((M, N))
    B = np.zeros((M, N))
    C = np.zeros((M, N))
    D = np.zeros((M, N))

    alpha = np.zeros((M, N))
    beta = np.zeros((M, N))
    
    # Finding U^(n+1/2)
    A[0:M, 0:N] = -nu / (2*dx**2) + U_old[0:M, 0:N] / (2*dx)
    B[0:M, 0:N] = 1 / dt + nu / (dx**2) - U_old[0:M, 0:N] / (2*dx)
    C[0:M, 0:N] = -nu / (2*dx**2)
    
    D[1:M-1, 1:N-1] = S_old[1:M-1, 1:N-1] / dt + 0.5*(
        nu*(S_old[1:M-1, 2:N] - 2*S_old[1:M-1, 1:N-1] + S_old[1:M-1, 0:N-2]) \
            / (dx**2) \
        - U_old[1:M-1, 1:N-1] * (S_old[1:M-1, 2:N] - S_old[1:M-1, 1:N-1]) \
            / dx) \
        + nu*(S_old[2:M, 1:N-1] - 2*S_old[1:M-1, 1:N-1] + S_old[0:M-2, 1:N-1]) \
            / (dy**2) \
        - V_old[1:M-1, 1:N-1] * (S_old[2:M, 1:N-1] - S_old[1:M-1, 1:N-1]) \
            / dy
    
    # Thomas algorithm for x
    # U(t, x=0, y) = 0
    alpha[0:M, 1] = 0
    beta[0:M, 1] = 0
        
    for i in range(1, N-1):
        alpha[1:M-1, i+1] = -A[1:N-1, i] \
            / (B[1:M-1, i] + C[1:N-1, i]*alpha[1:N-1, i])
        beta[1:M-1, i+1] = (D[1:M-1, i] - C[1:N-1, i]*beta[1:M-1, i]) \
            / (B[1:M-1, i] + C[1:N-1, i]*alpha[1:N-1, i])
        
    # U^(n+1/2)
    S_new[0:M, N-1] = S_old[0:M, N-1]
    for i in range(N-2, -1, -1):
        S_new[1:M-1, i] = alpha[1:M-1, i+1]*S_new[1:M-1, i+1] \
                            + beta[1:M-1, i+1]   
    
    # Finding U^(n+1)
    A[0:M, 0:N] = -nu / (2*dy**2) + V_old[0:M, 0:N] / (2*dy)
    B[0:M, 0:N] = 1 / dt + nu / (dy**2) - V_old[0:M, 0:N] / (2*dy)
    C[0:M, 0:N] = -nu / (2*dy**2)

    # U_new is U^(n+1/2) now
    D[1:M-1, 1:N-1] = S_new[1:M-1, 1:N-1] / dt - 0.5*(
        nu*(S_old[2:M, 1:N-1] - 2*S_old[1:M-1, 1:N-1] + S_old[0:M-2, 1:N-1]) \
            / (dy**2) \
        - V_old[1:M-1, 1:N-1] * (S_old[2:M, 1:N-1] - S_old[1:M-1, 1:N-1]) \
            / dy)

    # Thomas algorithm for y
    # Uy(t, 0<x<0.2, y=0) = 0
    alpha[1, 0:N1] = 1
    beta[1, 0:N1] = 0
    # U(t, 0.2<x<0.8, y=0) = 0
    alpha[1, N1:N4] = 0
    beta[1, N1:N4] = 0
    # Uy(t, 0.8<x<1, y=0) = 0
    alpha[1, N4:N] = 1
    beta[1, N4:N] = 0
    
    for j in range(1, M-1):
        alpha[j+1, 1:N-1] = -A[j, 1:N-1] \
            / (B[j, 1:N-1] + C[j, 1:N-1]*alpha[j, 1:N-1])
        beta[j+1, 1:N-1] = (D[j, 1:N-1] - C[j, 1:N-1]*beta[j, 1:N-1]) \
            / (B[j, 1:N-1] + C[j, 1:N-1]*alpha[j, 1:N-1])
        
    # U^(n+1)
    S_new[M-1, 0:N] = S_old[M-1, 0:N]
    for j in range(M-2, -1, -1):
        S_new[j, 1:N-1] = alpha[j+1, 1:N-1]*S_new[j+1, 1:N-1] \
                            + beta[j+1, 1:N-1]
    
    return S_new


# Returns numpy ndarray
# Passing by reference
@jit(nopython=True)
def Numba_ADM(S:np.ndarray, U:np.ndarray, V:np.ndarray, N, M, dx, dy, dt, nu) -> np.ndarray:
    U_old = U
    V_old = V
    S_old = S
    
    S_new = np.zeros_like(S)
    
    N1 = int(0.2 * N)
    N2 = int(0.3 * N)
    N3 = int(0.7 * N)
    N4 = int(0.8 * N)

    A = np.zeros((M, N))
    B = np.zeros((M, N))
    C = np.zeros((M, N))
    D = np.zeros((M, N))

    alpha = np.zeros((M, N))
    beta = np.zeros((M, N))
    
    # Finding U^(n+1/2)
    A[0:M, 0:N] = -nu / (dx**2) + U_old[0:M, 0:N] / (dx)
    B[0:M, 0:N] = 1 / dt + 2*nu / (dx**2) - U_old[0:M, 0:N] / (dx)
    C[0:M, 0:N] = -nu / (dx**2)
    
    D[1:M-1, 1:N-1] = S_old[1:M-1, 1:N-1] / dt \
        + nu*(S_old[2:M, 1:N-1] - 2*S_old[1:M-1, 1:N-1] + S_old[0:M-2, 1:N-1]) \
            / (dy**2) \
        - V_old[1:M-1, 1:N-1] * (S_old[2:M, 1:N-1] - S_old[1:M-1, 1:N-1]) \
            / dy
    
    # Thomas algorithm for x
    # U(t, x=0, y) = 0
    alpha[0:M, 1] = 0
    beta[0:M, 1] = 0
        
    for i in range(1, N-1):
        alpha[1:M-1, i+1] = -A[1:N-1, i] \
            / (B[1:M-1, i] + C[1:N-1, i]*alpha[1:N-1, i])
        beta[1:M-1, i+1] = (D[1:M-1, i] - C[1:N-1, i]*beta[1:M-1, i]) \
            / (B[1:M-1, i] + C[1:N-1, i]*alpha[1:N-1, i])
        
    # U^(n+1/2)
    S_new[0:M, N-1] = S_old[0:M, N-1]
    for i in range(N-2, -1, -1):
        S_new[1:M-1, i] = alpha[1:M-1, i+1]*S_new[1:M-1, i+1] \
                            + beta[1:M-1, i+1]   
    
    # Finding U^(n+1)
    A[0:M, 0:N] = -nu / (dy**2) + V_old[0:M, 0:N] / (dy)
    B[0:M, 0:N] = 1 / dt + 2*nu / (dy**2) - V_old[0:M, 0:N] / (dy)
    C[0:M, 0:N] = -nu / (dy**2)

    # U_new is U^(n+1/2) now
    D[1:M-1, 1:N-1] = S_new[1:M-1, 1:N-1] / dt \
        + nu*(S_new[1:M-1, 2:N] - 2*S_new[1:M-1, 1:N-1] + S_new[1:M-1, 0:N-2]) \
            / (dx**2) \
        - U_old[1:M-1, 1:N-1] * (S_new[1:M-1, 2:N] - S_new[1:M-1, 1:N-1]) \
            / dx

    # Thomas algorithm for y
    # Uy(t, 0<x<0.2, y=0) = 0
    alpha[1, 0:N1] = 1
    beta[1, 0:N1] = 0
    # U(t, 0.2<x<0.8, y=0) = 0
    alpha[1, N1:N4] = 0
    beta[1, N1:N4] = 0
    # Uy(t, 0.8<x<1, y=0) = 0
    alpha[1, N4:N] = 1
    beta[1, N4:N] = 0
    
    for j in range(1, M-1):
        alpha[j+1, 1:N-1] = -A[j, 1:N-1] \
            / (B[j, 1:N-1] + C[j, 1:N-1]*alpha[j, 1:N-1])
        beta[j+1, 1:N-1] = (D[j, 1:N-1] - C[j, 1:N-1]*beta[j, 1:N-1]) \
            / (B[j, 1:N-1] + C[j, 1:N-1]*alpha[j, 1:N-1])
        
    # U^(n+1)
    S_new[M-1, 0:N] = S_old[M-1, 0:N]
    for j in range(M-2, -1, -1):
        S_new[j, 1:N-1] = alpha[j+1, 1:N-1]*S_new[j+1, 1:N-1] \
                            + beta[j+1, 1:N-1]
                            
    return S_new

