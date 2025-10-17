import os
import time
import hashlib
import numpy as np

from src.utils import paillier, util, mesh_utils
from src.fast_deterministic_self_blinding.preprocessing import preprocess
from src.fast_deterministic_self_blinding.embedding import embed
from src.fast_deterministic_self_blinding.extracting import extract

def run(config: dict, encryption_keys: dict, watermarks: tuple, model):
    print("Run FDSB")
    signing_keys = util.genereate_signing_keys()
    keys = {"encryption": encryption_keys, "signing": signing_keys}
    
    result = {"config": config}
    
    encryptes_vertices, result_preprocess = preprocessing(model["vertices"], encryption_keys, config)
    result = result | result_preprocess

    signed_vertices, result_embedding = embedding(encrypted_vertices, watermarks[0], {"encryption": encryption_keys, "signing": signing_keys}, config)
    result = result | result_embedding

    decrypted_vertices, result_extracting = extracting(signed_vertices, {"encryption": encryption_keys, "signing": signing_keys}, config["quantisation_factor"], config["message_length"], config["self_blinding_length"], config["qim_step"])
    result = result | result_extracting

    recovered_mesh = recover_mesh(decrypted_vertices, encryption_keys["public"], config["quantisation_factor"])
    mesh_utils.save_3d_model(recovered_mesh, model["faces"], os.path.join(config["result_folder"], f"recovered_{config["model_name"]}"))

    result["BER_qim"] = util.compare_bits(watermarks[0], result["extracted_watermark"])

    return result


def recover_mesh(vertices: np.array, public_key: tuple, quantisation_factor: int) -> np.array:
    print("Vertices recovery")

    recovered_vertices = []
    N = public_key[0]

    for i_v in range(len(vertices)):
        for i_c in range(len(vertices[i_v])):
            if vertices[i_v][i_c] > N // 2:
                vertices[i_v][i_c] -= N
            vertices[i_v][i_c] = vertices[i_v][i_c] / (10**quantisation_factor)
    return vertices.astype(float)

def preprocessing(vertices: np.array, encryption_keys: dict, config: dict) -> (np.array, dict):
    return preprocess(vertices, encryption_keys, config)

def embedding(vertices: np.array, watermark: list, keys: dict, config: dict) -> (np.array, dict):
    return embed(vertices, watermark, keys, config)

def extracting(vertices: np.array, keys: dict, quantisation_factor: int, watermark_length: int, signature_length: int, qim_step: int) -> (np.array, dict):
    return extract(vertices, keys, quantisation_factor, watermark_length, signature_length, qim_step)