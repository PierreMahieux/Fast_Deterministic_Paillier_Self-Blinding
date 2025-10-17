import numpy as np
from gmpy2 import powmod, invert
from src.improved_robust_reversible_data_hiding.directions import (
    calculate_F_Nl, 
    calculate_T_Nl, 
    calculate_B_Nl,
    compute_all_directions_encrypted,
    calculate_direction_from_encrypted,
    get_M_vector
)

def extract_bit_from_direction(direction, F_Nl, T_Nl):
    threshold = F_Nl + T_Nl / 2
    
    if abs(direction) > threshold :
        return 1
    else:
        return 0


def decrypt_complete_model(encrypted_vertices, priv_key, pub_key):
    from src.improved_robust_reversible_data_hiding.encryption import decrypt_vertex
    import numpy as np
    
    decrypted_vertices = []
    
    for encrypted_vertex in encrypted_vertices:
        if encrypted_vertex is not None:
            decrypted_vertex = decrypt_vertex(encrypted_vertex, priv_key, pub_key)
            decrypted_vertices.append(decrypted_vertex)
    
    return np.array(decrypted_vertices)


