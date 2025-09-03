import random
import os

from src.high_capacity_data_hiding import hcdh
from src.utils import mesh_utils, util, paillier

BLOCK_SIZE = 31

if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)

    dataset_path = "../../datasets/meshes/"
    model_name = "casting.obj"
    model_path = os.path.join(script_dir, dataset_path + model_name)

    result_folder = os.path.join(script_dir, f"../../results/RDH/{model_name.split(".")[0]}/")
    result = {}

    model = mesh_utils.load_3d_model(os.path.join(script_dir, model_path))
    auth_message = "Copyright 2025 - LATIM" 
    auth_bits = util.generate_watermark_bits(auth_message)

    config = {"result_folder": result_folder, "model_path": model_path, "model_name": model_name, "block_size": BLOCK_SIZE}

    result_preprocess = hcdh.preprocess(model, config)