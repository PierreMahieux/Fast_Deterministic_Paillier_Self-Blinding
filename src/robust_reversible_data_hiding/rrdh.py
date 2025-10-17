import os
from gmpy2 import mpz, powmod, invert, gcd, mpz_random, random_state, mpz_urandomb, next_prime
import time
import datetime, hashlib
import numpy as np

from math import ceil

from src.utils import paillier, util, mesh_utils

def run(config: dict, encryption_keys:dict, watermark: list, model):
    print("Run RRDH")
    vertices = model["vertices"]
    faces = model["faces"]
    n_vertices = len(vertices)
    pub_key = encryption_keys["public"]
    priv_key = encryption_keys["secret"]
    N, g = pub_key

    result = {"config": config}

    # Preprocessing
    print("Pre-processing")
    vertices_prep, prep_info = preprocess_vertices(vertices, config["quantisation_factor"])

    #Patch division
    (patches, patch_indices), (isolated_coords, isolated_indices) = divide_into_patches(vertices_prep.copy(), faces)
    patch_info = get_patch_info(patches, isolated_coords)

    result["patch_number"] = len(patch_indices)
    result["max_vertices_per_patch"] = max([len(patch_indices[i]) for i in range(len(patch_indices))])
    result["min_vertices_per_patch"] = min([len(patch_indices[i]) for i in range(len(patch_indices))])
    
    # Encryption
    print("Encryption")
    start_encryption = time.time()
    encrypted_vertices = encrypt_vertices(vertices_prep, pub_key)
    result["time_encryption"] = time.time() - start_encryption
    
    # Embedding  
    print("Embedding")
    start_embedding = time.time()
    watermarked_encrypted_vertices = embed(encrypted_vertices.copy(), patch_indices, watermark, pub_key, config)
    result["time_embedding"] = time.time() - start_embedding


    # Restauration and decrypted watermarked model writing
    watermarked_decrypted_vertices = watermarked_encrypted_vertices.copy()
    for v_i in range(len(watermarked_decrypted_vertices)):
        for c_i in range(3):
            watermarked_decrypted_vertices[v_i][c_i] = int(paillier.decrypt_CRT(watermarked_encrypted_vertices[v_i][c_i], priv_key, pub_key))
    watermarked_restored = inverse_preprocess_vertices(watermarked_decrypted_vertices, prep_info)
    mesh_utils.save_3d_model(watermarked_restored, faces, os.path.join(config["result_folder"],"watermarked_restored.obj"))

    # Extraction
    print("Extraction")
    start_extraction = time.time()
    (watermarked_patches, watermarked_patch_indices), (isolated_coords, isolated_indices) = divide_into_patches(watermarked_encrypted_vertices, faces)

    extracted_watermark = extract(watermarked_encrypted_vertices, watermarked_patch_indices, encryption_keys, config)
    result["time_extraction"] = time.time() - start_extraction

    # Restoration
    print("Restoration")
    restored_encrypted_vertices = restore_encrypted_vertices(watermarked_encrypted_vertices.copy(), extracted_watermark, watermarked_patch_indices, encryption_keys["public"], config)
    
    restored_decrypted_vertices = restored_encrypted_vertices.copy()
    for v_i in range(len(restored_decrypted_vertices)):
        for c_i in range(3):
            restored_decrypted_vertices[v_i][c_i] = int(paillier.decrypt_CRT(restored_encrypted_vertices[v_i][c_i], priv_key, pub_key))
    restored_decrypted_vertices = inverse_preprocess_vertices(restored_decrypted_vertices, prep_info)

    mesh_utils.save_3d_model(restored_decrypted_vertices, faces, os.path.join(config["result_folder"],"fully_restored.obj"))

    result["BER"] = util.compare_bits(extracted_watermark, watermark)
    
    return result


def encrypt_vertices(vertices, public_key) -> np.array:
    encrypted_vertices = []
    for v in range(len(vertices)):
        enc = []
        for coord in range(3):
            enc.append(paillier.encrypt_given_r(int(vertices[v][coord]), public_key, 1))
        encrypted_vertices.append(enc)
    return np.array(encrypted_vertices)

def embed(vertices, patches, watermark, pub_key, config):
    N,g = pub_key
    
    for w_i, w_bit in enumerate(watermark):
        if w_bit == 0:
            continue

        patch = patches[w_i//3]
        axis = w_i % 3

        F_Nl = calculate_F_Nl(len(patch))
        T_Nl = calculate_T_Nl(len(patch), t=50)
        B_Nl = calculate_B_Nl(len(patch), F_Nl, T_Nl)
        B_Nl_c = paillier.encrypt_given_r(B_Nl, pub_key, 1)

        pos_enc_direction = mpz(1)
        neg_enc_direction = mpz(1)

        for v in patch[1:]:
            pos_enc_direction *= (vertices[v][axis] * invert(vertices[patch[0]][axis], N**2)) % N**2
            neg_enc_direction *= (vertices[patch[0]][axis] * invert(vertices[v][axis], N**2)) % N**2

        mapped_direction = map_encrypted_direction((pos_enc_direction, neg_enc_direction), 2*F_Nl + T_Nl, pub_key)

        if mapped_direction > 0:
            for k in patch[1:] : vertices[k][axis] = (vertices[k][axis] * B_Nl_c) % N**2
        elif mapped_direction < 0:
            vertices[patch[0]][axis] = (vertices[patch[0]][axis] * B_Nl_c) % N**2

    return vertices

def map_encrypted_direction(encrypted_directions, direction_limit, pub_key) -> int:
    pos_direction, neg_direction = encrypted_directions
    N, g = pub_key
    i = 0
    while i < direction_limit:
        i_c = powmod(g, i, N**2)
        if i_c == pos_direction % N**2:
            return i
        elif i_c == neg_direction % N**2:
            return -i
        i += 1
    return


def extract(vertices, patches, encryption_keys, config):
    extracted_watermark = []
    pub_key = encryption_keys["public"]
    N, g = pub_key

    for w_i in range(config["message_length"]):
        patch = patches[w_i//3]
        axis = w_i % 3

        F_Nl = calculate_F_Nl(len(patch))
        T_Nl = calculate_T_Nl(len(patch), t=50)
        B_Nl = calculate_B_Nl(len(patch), F_Nl, T_Nl)

        pos_enc_direction = mpz(1)
        neg_enc_direction = mpz(1)

        for v in patch[1:]:
            pos_enc_direction *= (vertices[v][axis] * invert(vertices[patch[0]][axis], N**2)) % N**2
            neg_enc_direction *= (vertices[patch[0]][axis] * invert(vertices[v][axis], N**2)) % N**2

        mapped_direction = map_encrypted_direction((pos_enc_direction, neg_enc_direction), 3*F_Nl + 2*T_Nl, pub_key)

        if -F_Nl < mapped_direction < F_Nl :
            extracted_watermark.append(0)
        else : 
            extracted_watermark.append(1)

    return extracted_watermark

def restore_encrypted_vertices(vertices, extracted_watermark, patch_indices, pub_key, config):
    N,g = pub_key
    for w_i in range(config["message_length"]):
        if extracted_watermark[w_i] == 0:
            continue
        patch = patch_indices[w_i//3]
        axis = w_i % 3

        F_Nl = calculate_F_Nl(len(patch))
        T_Nl = calculate_T_Nl(len(patch), t=5)
        B_Nl = calculate_B_Nl(len(patch), F_Nl, T_Nl)
        B_Nl_c = paillier.encrypt_given_r(B_Nl, pub_key, 1)

        pos_enc_direction = mpz(1)
        neg_enc_direction = mpz(1)

        for v in patch[1:]:
            pos_enc_direction *= (vertices[v][axis] * invert(vertices[patch[0]][axis], N**2)) % N**2
            neg_enc_direction *= (vertices[patch[0]][axis] * invert(vertices[v][axis], N**2)) % N**2

        mapped_direction = map_encrypted_direction((pos_enc_direction, neg_enc_direction), 3*F_Nl + 2*T_Nl, pub_key)

        if mapped_direction > F_Nl + T_Nl:
            for k in patch[1:] : vertices[k][axis] = (vertices[k][axis] * invert(B_Nl_c, N**2)) % N**2
        elif mapped_direction < -F_Nl - T_Nl:
            vertices[patch[0]][axis] = (vertices[patch[0]][axis] * invert(B_Nl_c, N**2)) % N**2

    return vertices

def inverse_preprocess_vertices(vertices_positive, preprocessing_info):
    k = preprocessing_info['k']
    
    vertices_normalized = _dequantization(vertices_positive, k)
    
    vertices = _denormalize_vertices(vertices_normalized, preprocessing_info['normalization_params'])
    
    return vertices

def _dequantization(vertices_quantified, k=4):
    vertices_int = vertices_quantified - 10**k

    vertices = vertices_int.astype(float) / (10**k)

    return vertices

def _denormalize_vertices(vertices_normalized, normalization_params):
    v_min = normalization_params['v_min']
    v_range = normalization_params['v_range']
    
    vertices = vertices_normalized * v_range + v_min
    
    return vertices

def compute_all_directions_encrypted(encrypted_patch, N):
    directions_encrypted = []
    
    for j in range(3):  # x, y, z
        Cd = compute_encrypted_direction(encrypted_patch, j, N)
        directions_encrypted.append(Cd)
    
    return directions_encrypted

def preprocess_vertices(vertices, k=4):
    preprocessing_info = {'k': k, 'normalize': False}

    vertices_positive, norm_params = _normalize_vertices(vertices)
    preprocessing_info['normalization_params'] = norm_params
    preprocessing_info['normalize'] = True
    vertices_quantified = _quantisation(vertices_positive, k)
    return vertices_quantified, preprocessing_info

def _normalize_vertices(vertices):
    v_min = np.min(vertices, axis=0)
    v_max = np.max(vertices, axis=0)
    
    v_range = v_max - v_min
    
    vertices_normalized = (vertices - v_min) / v_range
    
    normalization_params = {
        'v_min': v_min,
        'v_max': v_max,
        'v_range': v_range
    }
    
    return vertices_normalized, normalization_params

def _quantisation(vertices, k=4):
    vertices_work = vertices.copy()
    
    vertices_int = np.round(vertices_work * (10**k)).astype(int)
    vertices_quantified = vertices_int + 10**k

    return vertices_quantified

def _encrypt_vertex(vertex_coords, pub_key, r):
    encrypted_coords = []
    for coord in vertex_coords:
        c = paillier.encrypt_given_r(int(coord), pub_key, r)
        encrypted_coords.append(c)
    
    return encrypted_coords

def divide_into_patches(vertices, faces):
    n_vertices = len(vertices)
    adjacency = _build_adjacency_graph(faces, n_vertices)
    
    unclassified = set(range(n_vertices))
    classified = set()
    patches_indices = []
    isolated_indices = []
    
    while unclassified:
        seed = min(unclassified)
        
        patch_vertices = _get_k_ring_neighbors(seed, adjacency,k=2)
        patch_vertices = patch_vertices - classified
        
        if len(patch_vertices) >= 2:
            patch_idx = [seed] + sorted([v for v in patch_vertices if v != seed])
            patches_indices.append(patch_idx)
            
            classified.update(patch_vertices)
            unclassified -= patch_vertices
        else:
            isolated_indices.append(seed)
            unclassified.remove(seed)
            classified.add(seed)
    
    patches = []
    if patches_indices:
        for indices in patches_indices:
                patch = vertices[indices]
                patches.append(patch)
    
    isolated_coords = []
    if isolated_indices:
        for indices in isolated_indices:
            isolated_coords.append(vertices[indices])
            
    return (patches, patches_indices), (isolated_coords, isolated_indices)

def get_patch_info(patches, isolated_coords=None):
    sizes = [len(patch) for patch in patches]
    
    info = {
        'n_patches': len(patches),
        'min_size': min(sizes) if sizes else 0,
        'max_size': max(sizes) if sizes else 0,
        'avg_size': np.mean(sizes) if sizes else 0,
        'sizes': sizes,
        'n_isolated': len(isolated_coords) if isolated_coords else 0
    }
    
    return info

def _build_adjacency_graph(faces, n_vertices):
    adjacency = [set() for _ in range(n_vertices)]
    
    for face in faces:
        v1, v2, v3 = face[0], face[1], face[2]
        adjacency[v1].update([v2, v3])
        adjacency[v2].update([v1, v3])
        adjacency[v3].update([v1, v2])
    
    return adjacency

def _get_k_ring_neighbors(vertex_idx, adjacency, k=2):
    if k == 0:
        return {vertex_idx}
    
    neighbors = {vertex_idx}
    current_ring = {vertex_idx}
    
    for _ in range(k):
        next_ring = set()
        for v in current_ring:
            next_ring.update(adjacency[v])
        current_ring = next_ring - neighbors
        neighbors.update(next_ring)
    
    return neighbors

def calculate_F_Nl(Nl):
    F_Nl = 1.925*(Nl - 1)**3 - 60.6*(Nl - 1)**2 + 528*(Nl - 1) - 609

    return 20*F_Nl

def calculate_T_Nl(Nl, t=50):
    return t * (Nl - 1)

def calculate_B_Nl(Nl, F_Nl, T_Nl):    
    B_Nl = ceil((F_Nl + T_Nl) / (Nl - 1))
    return B_Nl