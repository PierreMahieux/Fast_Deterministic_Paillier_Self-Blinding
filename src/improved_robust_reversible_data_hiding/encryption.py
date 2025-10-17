from gmpy2 import mpz, powmod, invert
from src.utils import paillier

def encrypt_vertex(vertex_coords, pub_key, r):
    encrypted_coords = []
    for coord in vertex_coords:
        c = paillier.encrypt_given_r(int(coord), pub_key, r)
        encrypted_coords.append(c)
    
    return encrypted_coords


def encrypt_patch(patch, pub_key, r=None):
    N, g = pub_key
    
    if r is None:
        r = get_r(N)

    encrypted_patch = []
    for vertex in patch:
        encrypted_vertex = encrypt_vertex(vertex, pub_key, r)
        encrypted_patch.append(encrypted_vertex)
    
    return encrypted_patch, r


def encrypt_patches(patches, pub_key):
    N, g = pub_key
    encrypted_patches = []
    r_values = []
    
    for patch in patches:
        r = paillier.generate_r(N)
        encrypted_patch, _ = encrypt_patch(patch, pub_key, r)
        
        encrypted_patches.append(encrypted_patch)
        r_values.append(r)
    
    return encrypted_patches, r_values


def decrypt_vertex(encrypted_coords, priv_key, pub_key):
    decrypted_coords = []
    for c in encrypted_coords:
        m = paillier.decrypt_CRT(c, priv_key, pub_key)
        decrypted_coords.append(int(m))
    
    return decrypted_coords


def decrypt_patch(encrypted_patch, priv_key, pub_key):
    import numpy as np
    
    decrypted_patch = []
    for encrypted_vertex in encrypted_patch:
        decrypted_vertex = decrypt_vertex(encrypted_vertex, priv_key, pub_key)
        decrypted_patch.append(decrypted_vertex)
    
    return np.array(decrypted_patch)


def decrypt_patches(encrypted_patches, priv_key, pub_key):
    decrypted_patches = []
    
    for encrypted_patch in encrypted_patches:
        decrypted_patch = decrypt_patch(encrypted_patch, priv_key, pub_key)
        decrypted_patches.append(decrypted_patch)
    
    return decrypted_patches
