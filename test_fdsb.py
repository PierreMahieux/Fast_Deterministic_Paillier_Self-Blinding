import random
import os
import numpy as np

from src.fast_deterministic_self_blinding import fdsb, embedding, extracting, preprocessing
from src.utils import mesh_utils, util, paillier


QUANTISATION_FACTOR = 4
QIM_STEP = 4
KEY_SIZE = 512
LENGTH_SIGNATURE = 512
LENGTH_MESSAGE = 256

if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)

    dataset_path = "./datasets/meshes/"
    model_name = "casting.obj"
    model_path = os.path.join(script_dir, dataset_path + model_name)

    result_folder = os.path.join(script_dir, f"./results/fdsb/{model_name.split(".")[0]}/")
    result = {}

    model = mesh_utils.load_3d_model(os.path.join(script_dir, model_path))
    watermark_bits = [k%2 for k in range(LENGTH_MESSAGE)]
    random.shuffle(watermark_bits)

    encryption_keys = paillier.generate_keys(KEY_SIZE)
    signing_keys = util.genereate_signing_keys()

    config = {"key_size": KEY_SIZE, "quantisation_factor": QUANTISATION_FACTOR, "qim_step": QIM_STEP, "result_folder": result_folder, "model_path": model_path, "model_name": model_name, "signature_length": LENGTH_SIGNATURE, "message_length": LENGTH_MESSAGE}

    encrypted_vertices, result_preprocess = fdsb.preprocessing(model["vertices"], encryption_keys, config)
    result = result | result_preprocess

    signed_vertices, result_embedding = fdsb.embedding(encrypted_vertices, watermark_bits, {"encryption": encryption_keys, "signing": signing_keys}, config)
    result = result | result_embedding

    decrypted_vertices, result_extracting = fdsb.extracting(signed_vertices, {"encryption": encryption_keys, "signing": signing_keys}, config["quantisation_factor"], config["message_length"], config["signature_length"], config["qim_step"])
    result = result | result_extracting

    recovered_mesh = fdsb.recover_mesh(decrypted_vertices, encryption_keys["public"], config["quantisation_factor"])

    mesh_utils.save_3d_model(recovered_mesh, model["faces"], os.path.join(result_folder, f"recovered_{model_name}"))

    result["BER_qim"] = util.compare_bits(watermark_bits, result["extracted_watermark"])
    result["config"] = config

    util.write_report(result)

    print("fini")