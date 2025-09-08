import numpy as np
import sys

from datetime import datetime
import os

from src.utils import mesh_utils, util, paillier
from src.robust_reversible_data_hiding import rrdh

# from preprocessing import preprocess_vertices, inverse_preprocess_vertices
# from patch_division import divide_into_patches, get_patch_info
# from encryption import (
#     generate_keys_for_rrdh, encrypt_patches,encrypt_isolated_vertices
# )
# from watermarking import embed_watermark_in_model
# from extraction_restoration_ED import (
#     extract_watermark_from_model, restore_encrypted_patches_from_watermarking,
#     reconstruct_encrypted_model, decrypt_complete_model
# )
# from compare_visualization import compare_meshes
# from watermarking import calculate_ber

KEY_SIZE = 256
QUANTISATION_FACTOR = 4


if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)

    dataset_path = "./datasets/meshes/"
    model_name = "casting.obj"
    model_path = os.path.join(script_dir, dataset_path + model_name)

    model = mesh_utils.load_3d_model(model_path)
    
    vertices = model["vertices"]
    faces = model["faces"]
    n_vertices = len(vertices)
    result_folder = os.path.join(script_dir, f"./results/rrdh/{model_name.split(".")[0]}/")

    config = {"key_size": KEY_SIZE, "quantisation_factor": QUANTISATION_FACTOR, "result_folder": result_folder, "model_path": model_path, "model_name": model_name}

    result = {"config": config}

    # Preprocessing
    vertices_prep, prep_info = rrdh.preprocess_vertices(vertices, config["quantisation_factor"])

    # 2. DIVISION EN PATCHES
    print("\n2. Division en patches...")
    (patches, patch_indices), (isolated_coords, isolated_indices) = rrdh.divide_into_patches(vertices_prep, faces)
    patch_info = rrdh.get_patch_info(patches, isolated_coords)

    #génération des clés
    encryption_keys = paillier.generate_keys(config["key-size"])
    pub_key = encryption_keys["public"]
    priv_key = encryption_keys["secret"]
    N, g = pub_key

    # 3. CHIFFREMENT DES PATCHES
    print("\n3. Chiffrement des patches...")
    start_encryption = time.time()
    encrypted_patches, r_values = rrdh.encrypt_patches(patches, pub_key)
    encrypted_isolated = rrdh.encrypt_isolated_vertices(isolated_coords, pub_key) if isolated_coords else []
    #recosntruction du modèle chiffré complet
    encrypted_vertices = rrdh.recover_encrypted_model(
        encrypted_patches, patch_indices, encrypted_isolated, isolated_indices, n_vertices
    )
    result["time_encryption"] = time.time() - start_encryption
    
    # 4. GÉNÉRATION DU WATERMARK
    print("\n4. Génération du watermark...")
    watermark_length = patch_info['n_patches'] * 3
    watermark_original = [np.random.randint(0, 2) for _ in range(watermark_length)]

    
    # 5. TATOUAGE DANS LE DOMAINE CHIFFRÉ
    print("\n5. Tatouage dans le domaine chiffré...")
    start_embedding = time.time()
    watermarked_patches, nb_watermaked_bits = rrdh.embed_watermark_in_model(
        encrypted_patches, watermark_original, N, config["quantisation_factor"]
    )
    
    # Reconstruction des vertices chiffrés tatoués
    watermarked_encrypted_vertices = rrdh.recover_encrypted_model(
        watermarked_patches, patch_indices, encrypted_isolated, isolated_indices, n_vertices
    )
    result["time_embedding"] = start_embedding - time.time()
    
    # 6. EXTRACTION DANS LE DOMAINE CHIFFRÉ
    print("\n6. Extraction")
    start_extraction = time.time()
    extracted_watermark = rrdh.extract_watermark_from_model(
        watermarked_patches, N, 
        expected_length=watermark_length,
        k=config["quantisation_factor"]
    )
    result["time_extraction"] = time.time() - start_extraction
    # Calcul du BER
    ber = util.compare_bits(watermark_original, extracted_watermark)
    result["BER"] = ber

    
    # 7. RESTAURATION DANS LE DOMAINE CHIFFRÉ
    print("\n7. Restauration")
    
    restored_encrypted_patches = rrdh.restore_encrypted_patches_from_watermarking(watermarked_patches, N, config["quantisation_factor"])
    
    # Reconstruction des vertices chiffrés restaurés
    restored_encrypted_vertices = rrdh.recover_encrypted_model(
        restored_encrypted_patches, patch_indices, encrypted_isolated, isolated_indices, n_vertices
    )
    # Vertices restaurés déchiffrés
    restored_decrypted_vertices = rrdh.decrypt_complete_model(restored_encrypted_vertices, priv_key, pub_key)
    # Restauration compléte en appliquant l'inverse du preprocessing
    restored_clear = rrdh.inverse_preprocess_vertices(restored_decrypted_vertices, prep_info)
    
    # 8. Modèle déchiffré tatoué
    print("\n8. Modèle déchiffré tatoué...")
    
    watermarked_decrypted_vertices = rrdh.decrypt_complete_model(watermarked_encrypted_vertices, priv_key, pub_key)
    # Inverse preprocessing
    watermarked_clear = rrdh.inverse_preprocess_vertices(watermarked_decrypted_vertices, prep_info)


    # Sauvegarder les modèles
    # save_3d_model(vertices, faces, os.path.join(result_folder,"original.obj"))
    mesh_utils.save_3d_model(vertices_prep, faces, os.path.join(result_folder,"preprocessed.obj"))
    mesh_utils.save_3d_model(watermarked_clear, faces, os.path.join(result_folder,"watermarked_decrypted.obj"))
    mesh_utils.save_3d_model(restored_clear, faces, os.path.join(result_folder,"restored_decrypted.obj"))
    mesh_utils.save_3d_model(restored_decrypted_vertices, faces, os.path.join(result_folder,"restored_decrypted.obj"))
    
    # # RÉSUMÉ DES RÉSULTATS
    # report_path = os.path.join(result_folder, "rapport.txt")

    # f = open(report_path, 'w', encoding='utf-8')
    # f.write("RAPPORT RRDH-ED - TATOUAGE ROBUSTE ET RÉVERSIBLE\n")
    # f.write(f"Modèle: {os.path.basename(obj_filename)}\n")
    # f.write(f"Nombre de vertices: {n_vertices}\n")
    # f.write(f"Nombre de patches: {patch_info['n_patches']}, patch min: {patch_info['min_size']}, patch max: {patch_info['max_size']}\n")
    # f.write(f"keysize: {N.bit_length()}, g = N + 1: {g == N + 1}\n")
    # f.write(f"BER: {ber:.4f}\n")
    # f.write("\nDistances de Hausdorff:\n")
    
    # #Redirection temporaire de stdout
    # original_path = os.path.join(result_folder,"original.obj")
    # preprocess_path = os.path.join(result_folder,"preprocessed.obj")
    # watermarked_path = os.path.join(result_folder,"watermarked_decrypted.obj")
    # restored_path = os.path.join(result_folder,"restored_decrypted.obj")
    # restored_prep_path = os.path.join(result_folder,"restored_decrypted_prep.obj")
    # import sys
    # sys_stdout = sys.stdout
    # sys.stdout = f
    # print("1. Comparaison Preprocessed vs Watermarked:")
    # compare_meshes(preprocess_path, watermarked_path)
    # print("\n2. Comparaison Preprocessed vs Restored Preprocessed:")
    # compare_meshes(preprocess_path, restored_prep_path)
    # print("\n3. Comparaison Original vs Restored:")
    # compare_meshes(original_path, restored_path)

    # Restauration de stdout
    # sys.stdout = sys_stdout

    util.write_report(result)

    print("fini")