import numpy as np
import sys
import gmpy2
from datetime import datetime
from glob import glob
import time
import os

from src.utils import mesh_utils, util, paillier
from src.robust_reversible_data_hiding import rrdh

from gmpy2 import mpz, powmod, invert, gcd, mpz_random, random_state, mpz_urandomb, next_prime

KEY_SIZE = 512
QUANTISATION_FACTOR = 4
MESSAGE_LENGTH = 256

def complete_test():
    script_dir = os.path.dirname(__file__)

    dataset_path = "./datasets/meshes/"
    model_name = "casting.obj"
    model_path = os.path.join(script_dir, dataset_path + model_name)

    model = mesh_utils.load_3d_model(model_path)
    
    vertices = model["vertices"]
    faces = model["faces"]
    n_vertices = len(vertices)
    result_folder = os.path.join(script_dir, f"./results/rrdh/{model_name.split(".")[0]}/")

    config = {"key_size": KEY_SIZE, "quantisation_factor": QUANTISATION_FACTOR, "message_length": MESSAGE_LENGTH, "result_folder": result_folder, "model_path": model_path, "model_name": model_name}

    result = {"config": config}

    #génération des clés
    encryption_keys = paillier.generate_keys(config["key_size"])
    pub_key = encryption_keys["public"]
    priv_key = encryption_keys["secret"]
    config["encryption_keys"] = encryption_keys
    N, g = pub_key

    # Preprocessing
    print("Pre-processing")
    vertices_prep, prep_info = rrdh.preprocess_vertices(vertices, config["quantisation_factor"])
    mesh_utils.save_3d_model(vertices_prep, faces, os.path.join(result_folder,"preprocessed.obj"))

    #Patch division
    (patches, patch_indices), (isolated_coords, isolated_indices) = rrdh.divide_into_patches(vertices_prep.copy(), faces)
    patch_info = rrdh.get_patch_info(patches, isolated_coords)
    
    # watermark_original = [np.random.randint(0, 2) for _ in range(config["message_length"])]
    watermark = [1 for _ in range(config["message_length"])]

    # Encryption
    print("Encryption")
    start_encryption = time.time()
    encrypted_vertices = rrdh.encrypt_vertices(vertices_prep, pub_key)
    result["time_encryption"] = time.time() - start_encryption
    
    for i in range(len(encrypted_vertices)):
        for j in range(3):
            encrypted_vertices[i][j] = int(gmpy2.digits(encrypted_vertices[i][j])) % (10**config["quantisation_factor"])
    mesh_utils.save_3d_model(encrypted_vertices, model["faces"], os.path.join(result_folder, f"encrypted_{model_name}"))

    # Embedding  
    print("Embedding")
    start_embedding = time.time()
    watermarked_encrypted_vertices = rrdh.embed(encrypted_vertices.copy(), patch_indices, watermark, pub_key, config)
    result["time_embedding"] = time.time() - start_embedding


    # Restauration et sauvegarde du modèle tatoué déchiffré
    watermarked_decrypted_vertices = watermarked_encrypted_vertices.copy()
    for v_i in range(len(watermarked_decrypted_vertices)):
        for c_i in range(3):
            watermarked_decrypted_vertices[v_i][c_i] = int(paillier.decrypt_CRT(watermarked_encrypted_vertices[v_i][c_i], priv_key, pub_key))
    watermarked_restored = rrdh.inverse_preprocess_vertices(watermarked_decrypted_vertices, prep_info)
    mesh_utils.save_3d_model(watermarked_restored, faces, os.path.join(result_folder,"watermarked_restored.obj"))

    # Extraction
    print("Extraction")
    start_extraction = time.time()
    (watermarked_patches, watermarked_patch_indices), (isolated_coords, isolated_indices) = rrdh.divide_into_patches(watermarked_encrypted_vertices, faces)

    extracted_watermark = rrdh.extract(watermarked_encrypted_vertices, watermarked_patch_indices, encryption_keys, config)
    result["time_extraction"] = time.time() - start_extraction

    # Restoration
    print("Restoration")
    restored_encrypted_vertices = rrdh.restore_encrypted_vertices(watermarked_encrypted_vertices.copy(), extracted_watermark, watermarked_patch_indices, encryption_keys["public"], config)
    
    restored_decrypted_vertices = restored_encrypted_vertices.copy()
    for v_i in range(len(restored_decrypted_vertices)):
        for c_i in range(3):
            restored_decrypted_vertices[v_i][c_i] = int(paillier.decrypt_CRT(restored_encrypted_vertices[v_i][c_i], priv_key, pub_key))
    restored_decrypted_vertices = rrdh.inverse_preprocess_vertices(restored_decrypted_vertices, prep_info)

    mesh_utils.save_3d_model(restored_decrypted_vertices, faces, os.path.join(result_folder,"fully_restored.obj"))

    result["BER"] = util.compare_bits(extracted_watermark, watermark)
    
    util.write_report(result)
    print("fini")

def compute_max_direction_from_patches(patch_indices, vertices):
    # calcul d_max pour test
    set_Nl = set([len(patch_indices[k]) for k in range(len(patch_indices))])
    list_dMax = [(k, float('-inf')) for k in set_Nl]
    for patch in patch_indices:
        for axis in range(3):
            d = 0
            index = float('inf')
            for i, (l, v) in enumerate(list_dMax):
                if l==len(patch):
                    index = i
                    break
            # list_index = list_dMax.index((len(patch), float('-inf')))
            for k in patch[1:]:
                d += vertices[k][axis] - vertices[patch[0]][axis]
            if abs(d) > list_dMax[index][1]: list_dMax[index] = (len(patch), abs(d))
    return list_dMax

if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)

    complete_test()

    # dataset_path = "./datasets/meshes/"
    # meshes_list = glob(dataset_path + "*.obj")
    # # meshes_list = [dataset_path + "casting.obj"]

    # for model_path in meshes_list:
    #     model = mesh_utils.load_3d_model(os.path.join(script_dir, model_path))
    #     model_name = model_path.split('/')[-1]
        
    #     vertices = model["vertices"]
    #     faces = model["faces"]
    #     n_vertices = len(vertices)
    #     result_folder = os.path.join(script_dir, f"./results/rrdh/{model_name.split(".")[0]}/")

    #     config = {"key_size": KEY_SIZE, "quantisation_factor": QUANTISATION_FACTOR, "message_length": MESSAGE_LENGTH, "result_folder": result_folder, "model_path": model_path, "model_name": model_name}

    #     #Patch division
    #     (patches, patch_indices), (isolated_coords, isolated_indices) = rrdh.divide_into_patches(vertices, faces)
    #     patch_info = rrdh.get_patch_info(patches, isolated_coords)

    #     list_maximum_directions = compute_max_direction_from_patches(patch_indices, vertices)

    #     list_patches_size = set(len(p) for p in patch_indices)
    #     list_F_Nl = []
    #     list_T_Nl = []
    #     list_F_Nl_T_Nl = []
    #     list_2F_Nl_T_Nl = []
    #     for patch in patch_indices:
    #         Nl = len(patch)
    #         F_Nl = 1.925*(Nl - 1)**3 - 60.6*(Nl - 1)**2 + 528*(Nl - 1) - 609
    #         T_Nl = 50 * (Nl - 1)
    #         list_F_Nl.append(F_Nl)
    #         list_T_Nl.append(T_Nl)
    #         list_F_Nl_T_Nl.append(F_Nl + T_Nl)
    #         list_2F_Nl_T_Nl.append(2*F_Nl + T_Nl)

    #     results = {"mean_F_Nl": np.mean(list_F_Nl), "mean_T_Nl": np.mean(list_T_Nl), "mean_F_Nl_T_Nl": np.mean(list_F_Nl_T_Nl), "mean_2F_Nl_T_Nl": np.mean(list_2F_Nl_T_Nl), "std_F_Nl": np.std(list_F_Nl), "std_T_Nl": np.std(list_T_Nl), "std_F_Nl_T_Nl": np.std(list_F_Nl_T_Nl), "std_2F_Nl_T_Nl": np.std(list_2F_Nl_T_Nl),}

    #     filename = os.path.join(config["result_folder"], "report.txt")
    #     with open(filename, 'a') as file:
    #         for key, value in results.items():
    #             file.write(f"\"{key}\": {value},\n")

    print("fini")
