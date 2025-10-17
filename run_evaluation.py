import os
import random
import numpy as np
from glob import glob
import argparse
import json

from src.fast_deterministic_self_blinding import fdsb
from src.reversible_data_hiding import rdh
from src.robust_reversible_data_hiding import rrdh
from src.improved_robust_reversible_data_hiding import improved_rrdh
from src.high_capacity_data_hiding import hcdh
from src.utils import mesh_utils, util, paillier

KEY_SIZE = 512
LENGTH_SELF_BLINDING_WATERMARK = 512
MESSAGE_LENGTH = 256
QUANTISATION_FACTOR = 4
HCDH_BLOCK_SIZE = 31
HCDH_NUMBER_MESSAGES = 2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/test_fdsb.json")
    args = parser.parse_args()
    
    script_dir = os.path.dirname(__file__)

    dataset_path = "./datasets/meshes/"

    config_path = args.config_path
    config_file = open(config_path, 'r')
    config = json.loads(config_file.read())
    config_file.close()

    list_methods = []
    if config["methods"] == "all":
        list_methods.extend([improved_rrdh, rrdh, fdsb, hcdh, rdh])
    elif "improved_rrdh" in config["methods"]:
        list_methods.append(improved_rrdh)
    elif "rrdh" in config["methods"]:
        list_methods.append(rrdh)
    elif "fdsb" in config["methods"]:
        list_methods.append(fdsb)
    elif "rdh" in config["methods"]:
        list_methods.append(rdh)
    elif "hcdh" in config["methods"]:
        list_methods.append(hcdh)
    else: list_methods.append(fdsb)

    meshes_list = []
    if config["models"] == "all":
        meshes_list = glob(dataset_path + "*.obj")
    else: 
        for m in config["models"]:
            meshes_list.extend(glob(dataset_path + m))
    
    watermark_bits = [k%2 for k in range(config["message_length"])]
    random.shuffle(watermark_bits)
    self_blinding_bits = [random.randint(0, 2) for _ in range(config["self_blinding_length"])]
    watermarks = (watermark_bits, self_blinding_bits)

    encryption_keys = paillier.generate_keys(config["key_size"])
    
    for model_path in meshes_list:
        model = mesh_utils.load_3d_model(os.path.join(script_dir, model_path))
        model_name = model_path.split('/')[-1]
        
        for method in list_methods:
            result_folder = os.path.join(script_dir, f"results/{method.__name__.split('.')[-1]}/{model_name.split(".")[0]}/")
            for f in glob(result_folder + "*"):
                os.remove(f)

            config.update({"result_folder": result_folder, "model_path": model_path, "model_name": model_name, "length_self_blinding_watermark": LENGTH_SELF_BLINDING_WATERMARK, "block_size": HCDH_BLOCK_SIZE, "number_messages": HCDH_NUMBER_MESSAGES, "method_name": method.__name__})

            result = method.run(config, encryption_keys, watermarks, model)

            util.write_report(result)

            print(f"Evaluation method {method.__name__.split('.')[-1]} done.")
        