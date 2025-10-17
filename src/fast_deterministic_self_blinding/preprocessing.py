import time
import numpy as np

from gmpy2 import mpz

from src.utils import paillier

def preprocess(vertices: np.array, encryption_keys: dict, config: dict) -> (np.array, dict):
    print("Pre-processing")
    quantisation_factor = config["quantisation_factor"]
    qim_step = config["qim_step"]

    encrypted_vertices = np.zeros(vertices.shape, dtype=mpz)

    for i_v,_  in enumerate(vertices):
        for i_c, _ in enumerate(vertices[i_v]):
            encrypted_vertices[i_v][i_c] = np.floor(vertices[i_v][i_c] * (10**quantisation_factor)).astype(int)
            if (encrypted_vertices[i_v][i_c]//qim_step) % 2 == 0:
                encrypted_vertices[i_v][i_c] = int(qim_step * (encrypted_vertices[i_v][i_c]//qim_step) + (qim_step / 2))
            else:
                encrypted_vertices[i_v][i_c] = int(qim_step * (encrypted_vertices[i_v][i_c]//qim_step) - (qim_step / 2))
            encrypted_vertices[i_v][i_c] = encrypted_vertices[i_v][i_c] % encryption_keys["public"][0]

    start_time_encryption = time.time()
    vertices = paillier.encrypt_vertices(encrypted_vertices, encryption_keys["public"])
    time_encryption = time.time() - start_time_encryption

    return vertices, {"time_encryption": time_encryption}


def _encryption_preprocessing(vertices: np.array, N, quantisation_factor) -> np.array:
    vertices_positive = np.zeros(vertices.shape, dtype=mpz)

    # Quantisation
    quant_vertices = np.floor(vertices * (10**quantisation_factor)).astype(int)
    
    for i, _ in enumerate(vertices_positive):
        vertex = quant_vertices[i]
        for j, _ in enumerate(vertex):
            v = np.floor(vertex[j]).astype(int)
            vertices_positive[i][j] = v % N
    
    return np.array(vertices_positive)
