import numpy as np
import sys

from datetime import datetime
import time
import os

from src.utils import mesh_utils, util, paillier
from src.improved_robust_reversible_data_hiding import improved_rrdh

KEY_SIZE = 256
QUANTISATION_FACTOR = 4
MESSAGE_LENGTH = 256


if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)

    dataset_path = "./datasets/meshes/"
    model_name = "casting.obj"
    model_path = os.path.join(script_dir, dataset_path + model_name)

    model = mesh_utils.load_3d_model(model_path)
    
    vertices = model["vertices"]
    faces = model["faces"]
    n_vertices = len(vertices)
    result_folder = os.path.join(script_dir, f"./results/improved_rrdh/{model_name.split(".")[0]}/")

    #génération des clés
    encryption_keys = paillier.generate_keys(KEY_SIZE)
    pub_key = encryption_keys["public"]
    priv_key = encryption_keys["secret"]
    N, g = pub_key

    config = {"key_size": KEY_SIZE, "quantisation_factor": QUANTISATION_FACTOR, "result_folder": result_folder, "message_length": MESSAGE_LENGTH, "model_path": model_path, "model_name": model_name, "encryption_keys": encryption_keys}

    result = {"config": config}

    # Preprocessing
    print("Pre-processing")
    vertices_prep, prep_info = improved_rrdh.preprocess_vertices(vertices, config["quantisation_factor"])

    # 2. DIVISION EN PATCHES
    (patches, patch_indices), (isolated_coords, isolated_indices) = improved_rrdh.divide_into_patches(vertices_prep, faces)
    patch_info = improved_rrdh.get_patch_info(patches, isolated_coords)


    # 3. CHIFFREMENT DES PATCHES
    print("Encryption")
    start_encryption = time.time()
    encrypted_patches, r_values = improved_rrdh.encrypt_patches(patches, pub_key)
    encrypted_isolated = improved_rrdh.encrypt_isolated_vertices(isolated_coords, pub_key) if isolated_coords else []
    #recosntruction du modèle chiffré complet
    encrypted_vertices = improved_rrdh.recover_encrypted_model(
        encrypted_patches, patch_indices, encrypted_isolated, isolated_indices, n_vertices
    )
    # encrypted_vertices = improved_rrdh.encrypt_vertices(vertices_prep, pub_key)

    result["time_encryption"] = time.time() - start_encryption
    
    # 4. GÉNÉRATION DU WATERMARK
    
    # watermark = [np.random.randint(0, 2) for _ in range(config["message_length"])]
    watermark = [1 for _ in range(config["message_length"])]

    
    # 5. TATOUAGE DANS LE DOMAINE CHIFFRÉ
    print("Embedding")
    start_embedding = time.time()
    watermarked_encrypted_vertices = improved_rrdh.embed(encrypted_vertices, patch_indices, watermark, pub_key, config)
    result["time_embedding"] = time.time() - start_embedding

    # Restauration et sauvegarde du modèle tatoué déchiffré
    watermarked_decrypted_vertices = np.array(watermarked_encrypted_vertices.copy())
    for v_i in range(len(watermarked_decrypted_vertices)):
        for c_i in range(3):
            watermarked_decrypted_vertices[v_i][c_i] = int(paillier.decrypt_CRT(watermarked_encrypted_vertices[v_i][c_i], priv_key, pub_key))
    watermarked_restored = improved_rrdh.inverse_preprocess_vertices(watermarked_decrypted_vertices, prep_info)
    mesh_utils.save_3d_model(watermarked_restored, faces, os.path.join(result_folder,"watermarked_restored.obj"))

    
    # 6. EXTRACTION DANS LE DOMAINE CHIFFRÉ
    print("Extraction")
    start_extraction = time.time()
    (watermarked_patches, watermarked_patch_indices), (isolated_coords, isolated_indices) = improved_rrdh.divide_into_patches(np.array(watermarked_encrypted_vertices), faces)

    extracted_watermark = improved_rrdh.extract(watermarked_encrypted_vertices, watermarked_patch_indices, encryption_keys, config)
    result["time_extraction"] = time.time() - start_extraction

    # Restoration
    print("Restoration")
    start_restoration = time.time()
    restored_encrypted_vertices = improved_rrdh.restore_encrypted_vertices(watermarked_encrypted_vertices.copy(), extracted_watermark, watermarked_patch_indices, encryption_keys["public"], config)
    result["time_restoration"] = time.time() - start_restoration

    restored_decrypted_vertices = np.array(restored_encrypted_vertices.copy())
    for v_i in range(len(restored_decrypted_vertices)):
        for c_i in range(3):
            restored_decrypted_vertices[v_i][c_i] = int(paillier.decrypt_CRT(restored_encrypted_vertices[v_i][c_i], priv_key, pub_key))
    restored_decrypted_vertices = improved_rrdh.inverse_preprocess_vertices(restored_decrypted_vertices, prep_info)

    mesh_utils.save_3d_model(restored_decrypted_vertices, faces, os.path.join(result_folder,"fully_restored.obj"))

    # Calcul du BER
    result["BER"] = util.compare_bits(watermark, extracted_watermark)
    util.write_report(result)

    print("fini")