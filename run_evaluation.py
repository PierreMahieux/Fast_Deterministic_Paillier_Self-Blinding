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
LENGTH_SIGNATURE = 512
MESSAGE_LENGTH = 521
QUANTISATION_FACTOR = 4

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
        list_methods.extend([fdsb, rdh])
    elif "methods" not in config or len(config["methods"]) < 1:
        list_methods.append(fdsb)
    else:
        if "fdsb" in config["methods"]:
            list_methods.append(fdsb)
        if "rdh" in config["methods"]:
            list_methods.append(rdh)
        if "rrdh" in config["methods"]:
            list_methods.append(rrdh)
        if "improved_rrdh" in config["methods"]:
            list_methods.append(improved_rrdh)
        if "hcdh" in config["methods"]:
            list_methods.append(hcdh)
    

    meshes_list = []
    if config["models"] == "all":
        meshes_list = glob(dataset_path + "*.obj")
    else: 
        for m in config["models"]:
            meshes_list.extend(glob(dataset_path + m))

    if "message_length" not in config:
        original_message_length = MESSAGE_LENGTH
    else:
        original_message_length = config["message_length"]
    if "self_blinding_length" not in config:
        original_self_blinding_length = LENGTH_SIGNATURE
    else:
        original_self_blinding_length = config["self_blinding_length"]
    
    encryption_keys = paillier.generate_keys(config["key_size"])
    
    for model_path in meshes_list:
        model = mesh_utils.load_3d_model(os.path.join(script_dir, model_path))
        model_name = model_path.split('/')[-1]


        if "list_qim_step" in config:
            for delta in config["list_qim_step"]:
                config["qim_step"] = delta
        
                for method in list_methods:
                    if "message_length" not in config:
                        config["message_length"] = MESSAGE_LENGTH
                    elif original_message_length == "max":
                        config["message_length"] = len(model["vertices"]) * 3
                    watermark_bits = [k%2 for k in range(config["message_length"])]
                    random.shuffle(watermark_bits)

                    if "self_blinding_length" not in config:
                        config["self_blinding_length"] = LENGTH_SIGNATURE
                    elif original_self_blinding_length == "max":
                        config["self_blinding_length"] = len(model["vertices"]) * 3
                    self_blinding_bits = [k%2 for k in range(config["self_blinding_length"])]
                    random.shuffle(self_blinding_bits)
                    watermarks = (watermark_bits, self_blinding_bits)

                    result_folder = os.path.join(script_dir, f"results/{method.__name__.split('.')[-1]}/{model_name.split(".")[0]}/delta_{config["qim_step"]}/")
                    for f in glob(result_folder + "*"):
                        os.remove(f)

                    config.update({"result_folder": result_folder, "model_path": model_path, "model_name": model_name, "signature_length": LENGTH_SIGNATURE, "method_name": method.__name__})

                    result = method.run(config, encryption_keys, watermarks, model)

                    util.write_report(result)

                    print(f"Evaluation method {method.__name__.split('.')[-1]} with model {model_name} and delta_qim {config["qim_step"]} done.")
        
    print("\nÉvaluation complète")