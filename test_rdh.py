import random
import os

from src.reversible_data_hiding import rdh
from src.utils import mesh_utils, util, paillier


QUANTISATION_FACTOR = 4
KEY_SIZE = 512

if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)

    dataset_path = "datasets/meshes/"
    model_name = "casting.obj"
    model_path = os.path.join(script_dir, dataset_path + model_name)

    result_folder = os.path.join(script_dir, f"./results/rdh/{model_name.split(".")[0]}/")
    result = {}

    model = mesh_utils.load_3d_model(os.path.join(script_dir, model_path))
    auth_message = "Copyright 2025 - LATIM" 
    auth_bits = util.hash_string_to_bits(auth_message)
    num_mgmt_bits = 512
    mgmt_bits=[random.randint(0, 1) for _ in range(num_mgmt_bits)]
    watermarks = (auth_bits, mgmt_bits)

    encryption_keys = paillier.generate_keys(KEY_SIZE)


    config = {"key_size": KEY_SIZE, "quantisation_factor": QUANTISATION_FACTOR, "result_folder": result_folder, "model_path": model_path, "model_name": model_name}

    encrypted_vertices, result_preprocess = rdh.preprocess(model, encryption_keys, config)
    result = result | result_preprocess

    embedded_vertices, result_embed = rdh.embed(encrypted_vertices, watermarks, encryption_keys["public"], config)
    result = result | result_embed

    result_extract = rdh.extract(embedded_vertices, encryption_keys, config["quantisation_factor"], (len(watermarks[0]), len(watermarks[1])))
    result = result | result_extract

    recovered_mesh = rdh.recover_mesh(result_extract["decrypted_vertices"], encryption_keys["public"], config["quantisation_factor"])
    mesh_utils.save_3d_model(recovered_mesh, model["faces"], os.path.join(result_folder, f"recovered_{model_name}"))

    result["BER_histogram_shifting"] = util.compare_bits(watermarks[0], result["histogram_shifting_watermark"])
    result["BER_self_blinding"] = util.compare_bits(watermarks[1], result["self_blinding_watermark"])
    result["config"] = config

    util.write_report(result)

    print("fini")