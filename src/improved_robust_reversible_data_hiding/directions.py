import numpy as np
from gmpy2 import mpz, powmod, invert
from math import ceil


def get_M_vector(Nl):
    M = np.ones(Nl, dtype=int)
    M[0] = -1
    return M


def calculate_F_Nl(Nl, k=4):
    # F_Nl = 1.925 * (Nl-1)**3 - 60.6 * (Nl-1)**2 + 528 * (Nl-1) - 609
    # # F(Nl) > d_max with security margin
    # F_Nl =  2 * F_Nl

    F_Nl = 2*(Nl - 1)*10**k

    return F_Nl

def calculate_T_Nl(Nl, t=50):
    return t * (Nl - 1)

def calculate_B_Nl(Nl, F_Nl, T_Nl, t=50, k=4):
    B_Nl = ceil((F_Nl + T_Nl) / (Nl - 1))
    return B_Nl

def compute_encrypted_direction(encrypted_patch, j, N):
    N2 = N * N
    Nl = len(encrypted_patch)
    M = get_M_vector(Nl)
    
    Cd = mpz(1)
    
    for i in range(Nl):
        coord_encrypted = encrypted_patch[i][j]
        
        if M[i] == 1:
            Cd = (Cd * coord_encrypted) % N2
        else:  # M[i] == -1
            C_inv = invert(coord_encrypted, N2)
            C_inv_power = powmod(C_inv, Nl-1, N2)
            Cd = (Cd * C_inv_power) % N2
    
    return Cd

def calculate_direction_from_encrypted(Cd, N, F_limit):
    d_mod_N = ((Cd - 1) // N) % N
    
    if d_mod_N <= F_limit and d_mod_N >= 0:
        d = int(d_mod_N)
    elif d_mod_N >= N - F_limit and d_mod_N <= N-1:
        d = int(d_mod_N) - N
    else:
        d = 0
        raise ValueError("Computed direction out of bound.")
    return d


def compute_all_directions_encrypted(encrypted_patch, N):
    directions_encrypted = []
    
    for j in range(3):
        Cd = compute_encrypted_direction(encrypted_patch, j, N)
        directions_encrypted.append(Cd)
    
    return directions_encrypted


def determine_directions_from_encrypted(directions_encrypted, N, Nl):
    F_Nl = calculate_F_Nl(Nl)
    directions = []
    
    for Cd in directions_encrypted:
        d = calculate_direction_from_encrypted(Cd, N, F_Nl)
        directions.append(d)
    
    return directions
