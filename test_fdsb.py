import random
import os

from src.fast_deterministic_self_blinding import fdsb
from src.utils import mesh_utils, util, paillier


QUANTISATION_FACTOR = 5
QIM_STEP = 4
KEY_SIZE = 2048
LENGTH_SIGNATURE = 512

if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)

    dataset_path = "./datasets/meshes/"
    model_name = "tumored_kidney.obj"
    model_path = os.path.join(script_dir, dataset_path + model_name)

    result_folder = os.path.join(script_dir, f"./results/RDH/{model_name.split(".")[0]}/")
    result = {}

    model = mesh_utils.load_3d_model(os.path.join(script_dir, model_path))
    qim_message = "Copyright 2025 - LATIM" 
    qim_bits = util.hash_string_to_bits(qim_message)
    watermarks = {"qim": qim_bits}

    encryption_keys = paillier.generate_keys(KEY_SIZE)
    signing_keys = util.genereate_signing_keys()

    config = {"key_size": KEY_SIZE, "quantisation_factor": QUANTISATION_FACTOR, "qim_step": QIM_STEP, "result_folder": result_folder, "model_path": model_path, "model_name": model_name, "length_signature": LENGTH_SIGNATURE}

    result_preprocess = fdsb.preprocess(model, encryption_keys, config)
    result = result | result_preprocess

    result_embedding = fdsb.embed(result_preprocess["encrypted_vertices"], watermarks, {"encryption": encryption_keys, "signing": signing_keys}, config)

    result_extracting = fdsb.extract(result_embedding["signed_vertices"], {"encryption": encryption_keys, "signing": signing_keys}, config["quantisation_factor"], (len(watermarks["qim"]), config["length_signature"]))

    print("fini")