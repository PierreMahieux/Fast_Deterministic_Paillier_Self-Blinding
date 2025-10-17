import numpy as np
import sys

import os
import time
from gmpy2 import mpz, powmod, invert
from src.improved_robust_reversible_data_hiding.preprocessing import preprocess_vertices, inverse_preprocess_vertices
from src.improved_robust_reversible_data_hiding.patch_division import divide_into_patches, get_patch_info
from src.improved_robust_reversible_data_hiding.encryption import (
    encrypt_patches, encrypt_vertex
)
from src.improved_robust_reversible_data_hiding.extraction_restoration_ED import (
    decrypt_complete_model, extract_bit_from_direction
)
from src.improved_robust_reversible_data_hiding import directions 
from src.utils import paillier, util, mesh_utils

def run(config: dict, encryption_keys:dict, watermark: list, model):
    print("Run Bamba RRDH")
    pub_key = encryption_keys["public"]
    priv_key = encryption_keys["secret"]

    N, g = pub_key

    vertices = model["vertices"]
    faces = model["faces"]
    n_vertices = len(vertices)

    result = {"config": config}

    # Preprocessing
    vertices_prep, prep_info = preprocess_vertices(vertices, k=4)
    # mesh_utils.save_3d_model(vertices_prep, faces, os.path.join(config["result_folder"],"preprocessed.obj"))

    # Patch dividing
    print("\nPatch dividing")
    (patches, patch_indices), (isolated_coords, isolated_indices) = divide_into_patches(vertices_prep, faces)
    patch_info = get_patch_info(patches, isolated_coords)

    result["patch_number"] = len(patch_indices)
    result["max_vertices_per_patch"] = max([len(patch_indices[i]) for i in range(len(patch_indices))])
    result["min_vertices_per_patch"] = min([len(patch_indices[i]) for i in range(len(patch_indices))])
    
    # Encryption
    print("\nEncryption")
    
    encrypted_patches, r_values = encrypt_patches(patches, pub_key)

    encrypted_vertices = []
    for v in vertices_prep:
        encrypted_vertices.append(encrypt_vertex(v, pub_key, 1)) # r fixe
    
    # Embedding
    print("\nWatermarking")
    start_embedding = time.time()

    watermarked_patches = encrypted_patches.copy()

    watermarked_encrypted_vertices = encrypted_vertices.copy()
    for w_i in range(len(watermark)):
        if watermark[w_i] == 0:
            continue
        
        patch = patch_indices[w_i//3]
        patch_vertices = [encrypted_vertices[i] for i in patch]
        axis = w_i % 3
        
        Nl = len(patch)
        F_Nl = directions.calculate_F_Nl(Nl, prep_info['k'])
        T_Nl = directions.calculate_T_Nl(Nl)
        B_Nl = directions.calculate_B_Nl(Nl, F_Nl, T_Nl, t=50, k=prep_info['k'])
        params = {
            'F_Nl': F_Nl,
            'T_Nl': T_Nl,
            'B_Nl': B_Nl
        }

        encrypted_direction = directions.compute_encrypted_direction(patch_vertices, axis, N)
        direction = directions.calculate_direction_from_encrypted(encrypted_direction, N, params["F_Nl"])

        g_B = powmod(g, params["B_Nl"], N**2)

        if direction >= 0:
            for k in patch[1:]:
                watermarked_encrypted_vertices[k][axis] = (watermarked_encrypted_vertices[k][axis] * g_B) % N**2
        elif direction < 0:
            watermarked_encrypted_vertices[patch[0]][axis] = (watermarked_encrypted_vertices[patch[0]][axis] * g_B) % N**2
        
    result["time_embedding"] = time.time() - start_embedding
    
    # Extraction
    print("\nExtraction")
    start_extraction = time.time()
    extracted_watermark = []
    for w_i in range(config["message_length"]):
        
        patch = patch_indices[w_i//3]
        patch_vertices = [watermarked_encrypted_vertices[i] for i in patch]
        axis = w_i % 3

        Nl = len(patch)
        F_Nl = directions.calculate_F_Nl(Nl,config["quantisation_factor"])
        T_Nl = directions.calculate_T_Nl(Nl)

        F_limit = 2 * F_Nl + T_Nl

        encrypted_direction = directions.compute_encrypted_direction(patch_vertices, axis, N)
        direction = directions.calculate_direction_from_encrypted(encrypted_direction, N, F_limit)
        extracted_watermark.append(extract_bit_from_direction(direction, F_Nl, T_Nl))
        
    result["time_extraction"] = time.time() - start_extraction
    
    result["BER"] = util.compare_bits(watermark, extracted_watermark)

    
    # Restoration
    print("\nRestoration")

    restored_encrypted_vertices = watermarked_encrypted_vertices.copy()
    for w_i in range(len(extracted_watermark)):
        if extracted_watermark[w_i] == 0:
            continue

        patch = patch_indices[w_i//3]
        patch_vertices = [restored_encrypted_vertices[i] for i in patch]
        axis = w_i % 3

        Nl = len(patch)
        F_Nl = directions.calculate_F_Nl(Nl,config["quantisation_factor"])
        T_Nl = directions.calculate_T_Nl(Nl)
        B_Nl = directions.calculate_B_Nl(Nl, F_Nl, T_Nl, t=50, k=config["quantisation_factor"])
        
        g = N + 1
        
        g_B = powmod(g, B_Nl, N**2)
        theta_g_B = invert(g_B, N**2)

        F_limit = 2 * F_Nl + T_Nl
        encrypted_direction = directions.compute_encrypted_direction(patch_vertices, axis, N)
        direction = directions.calculate_direction_from_encrypted(encrypted_direction, N, F_limit)
        
        threshold = F_Nl + T_Nl / 2

        if direction >= 0:
            for k in patch[1:]:
                restored_encrypted_vertices[k][axis] = (restored_encrypted_vertices[k][axis] * theta_g_B) % N**2
        else :
            restored_encrypted_vertices[patch[0]][axis] = (restored_encrypted_vertices[patch[0]][axis] * theta_g_B) % N**2

    restored_decrypted_vertices = decrypt_complete_model(restored_encrypted_vertices, priv_key, pub_key)
    
    restored_clear = inverse_preprocess_vertices(restored_decrypted_vertices, prep_info)

    mesh_utils.save_3d_model(restored_clear, faces, os.path.join(config["result_folder"],"fully_restored.obj"))
        
    watermarked_decrypted_vertices = np.array(watermarked_encrypted_vertices.copy())
    for v_i in range(len(watermarked_decrypted_vertices)):
        for c_i in range(3):
            watermarked_decrypted_vertices[v_i][c_i] = int(paillier.decrypt_CRT(watermarked_encrypted_vertices[v_i][c_i], priv_key, pub_key))
    
    watermarked_clear = inverse_preprocess_vertices(watermarked_decrypted_vertices, prep_info)
    mesh_utils.save_3d_model(watermarked_clear, faces, os.path.join(config["result_folder"],"watermarked_restored.obj"))
    
    return result