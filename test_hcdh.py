import random
import os
import time

from src.high_capacity_data_hiding import hcdh
from src.utils import mesh_utils, util, paillier

BLOCK_SIZE = 15
NUMBER_MESSAGES = 1

if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)

    dataset_path = "./datasets/meshes/"
    model_name = "casting.obj"
    model_path = os.path.join(script_dir, dataset_path + model_name)

    result_folder = os.path.join(script_dir, f"./results/hcdh/{model_name.split(".")[0]}/")
    result = {}

    model = mesh_utils.load_3d_model(os.path.join(script_dir, model_path))
    auth_message = "Copyright 2025 - LATIM" 
    auth_bits = util.hash_string_to_bits(auth_message)

    messages = [auth_bits for _ in range(NUMBER_MESSAGES)]

    config = {"result_folder": result_folder, "model_path": model_path, "model_name": model_name, "block_size": BLOCK_SIZE, "number_messages": NUMBER_MESSAGES}

    result_preprocess = hcdh.preprocessing(model, config)
    params = result_preprocess["parameters"]
    params["key_size"] = result_preprocess["encryption_keys"]["public"][0].bit_length()
    result = result | result_preprocess

    start_encryption = time.time()
    encrypted_blocks_data = hcdh.encrypt_all_blocks(result_preprocess)
    result["time_encryption"] = time.time() - start_encryption

    encrypted_vertices = hcdh.reconstruct_watermarked_vertices(encrypted_blocks_data)
    mesh_utils.save_3d_model(encrypted_vertices, model["faces"], os.path.join(result_folder, f"encrypted_{model_name}"))

    start_embedding = time.time()
    alpha = params['alpha']
    watermarked_data = hcdh.embedding(encrypted_blocks_data, messages, alpha)
    result["time_embedding"] = time.time() - start_embedding

    start_extraction = time.time()
    extracted_messages = hcdh.extracting(watermarked_data)
    result["time_extraction"] = time.time() - start_extraction

    decrypted_vertices = hcdh.reconstruct_decrypt_vertices(encrypted_blocks_data)
    mesh_utils.save_3d_model(decrypted_vertices, model["faces"], os.path.join(result_folder,"restored_decrypted.obj"))

    for i in range(config["number_messages"]):
        result[f"ber_{i}"] = util.compare_bits(messages[i], extracted_messages[i])
        print(result[f"ber_{i}"])

    result["config"] = config

    util.write_report(result)

    print("fini")