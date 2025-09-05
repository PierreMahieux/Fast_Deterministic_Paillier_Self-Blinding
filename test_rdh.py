import random
import os

from src.reversible_data_hiding import rdh
from src.utils import mesh_utils, util, paillier


QUANTISATION_STEP = 4
KEY_SIZE = 2048

if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)

    dataset_path = "../../datasets/meshes/"
    model_name = "casting.obj"
    model_path = os.path.join(script_dir, dataset_path + model_name)

    result_folder = os.path.join(script_dir, f"../../results/RDH/{model_name.split(".")[0]}/")
    result = {}

    model = mesh_utils.load_3d_model(os.path.join(script_dir, model_path))
    auth_message = "Copyright 2025 - LATIM" 
    auth_bits = util.hash_string_to_bits(auth_message)
    num_mgmt_bits = 8
    mgmt_bits=[random.randint(0, 1) for _ in range(num_mgmt_bits)]
    watermarks = {"histogram_shifting": auth_bits, "self_blinding": mgmt_bits}

    encryption_keys = paillier.generate_keys(KEY_SIZE)


    config = {"key_size": KEY_SIZE, "quantisation_step": QUANTISATION_STEP, "result_folder": result_folder, "model_path": model_path, "model_name": model_name}

    result_preprocess = rdh.preprocess(model, encryption_keys, config)
    result = result | result_preprocess

    result_embed = rdh.embed(result_preprocess["encrypted_vertices"], watermarks, encryption_keys["public"], config)
    embedded_vertices = result_embed["embedded_encrypted_model"]
    mesh_utils.save_3d_model(result_preprocess["quantified_model"], model["faces"], os.path.join(config["result_folder"], f"quantified_{config["model_name"]}"))
    result = result | result_embed

    result_extract = rdh.extract(embedded_vertices, encryption_keys, config["quantisation_step"], (len(watermarks["histogram_shifting"]), len(watermarks["self_blinding"])))
    result = result | result_extract

    recovered_mesh = rdh.recover_mesh(result_extract["decrypted_vertices"], encryption_keys["public"], config["quantisation_step"])
    mesh_utils.save_3d_model(recovered_mesh, model["faces"], os.path.join(result_folder, f"recovered_{model_name}"))

    result["BER_histogram_shifting"] = util.compare_bits(watermarks["histogram_shifting"], result["histogram_shifting_watermark"])
    result["BER_self_blinding"] = util.compare_bits(watermarks["self_blinding"], result["self_blinding_watermark"])
    result["config"] = config

    util.write_report(result)

    print("fini")