import random
import os

from src.fast_deterministic_self_blinding import fdsb
from src.utils import mesh_utils, util, paillier


QUANTISATION_DELTA = 4
QIM_STEP = 2
KEY_SIZE = 512

if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)

    dataset_path = "./datasets/meshes/"
    model_name = "tumored_brain.obj"
    model_path = os.path.join(script_dir, dataset_path + model_name)

    result_folder = os.path.join(script_dir, f"./results/RDH/{model_name.split(".")[0]}/")
    result = {}

    model = mesh_utils.load_3d_model(os.path.join(script_dir, model_path))
    auth_message = "Copyright 2025 - LATIM" 
    auth_bits = util.generate_watermark_bits(auth_message)
    num_mgmt_bits = 256
    mgmt_bits=[random.randint(0, 1) for _ in range(num_mgmt_bits)]
    watermarks = {"qim": auth_bits, "self_blinding": mgmt_bits}

    encryption_keys = paillier.generate_keys(KEY_SIZE)

    config = {"key_size": KEY_SIZE, "quantisation_delta": QUANTISATION_DELTA, "qim_step": QIM_STEP, "result_folder": result_folder, "model_path": model_path, "model_name": model_name, }

    result_preprocess = fdsb.preprocess(model, encryption_keys, config)
    result = result | result_preprocess

    result_embedding = fdsb.embed(result_preprocess["encrypted_vertices"], watermarks, encryption_keys, config)

    print("fini")