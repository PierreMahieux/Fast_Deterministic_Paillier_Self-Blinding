import time
import numpy as np

from gmpy2 import mpz, powmod

from src.utils import paillier

def preprocess(vertices: np.array, encryption_keys: dict, config: dict) -> (np.array, dict):
    print("Pre-processing")
    quantisation_factor = config["quantisation_factor"]
    qim_step = config["qim_step"]

    quantized = np.array([[mpz(np.floor(vertices[i_v][i_c] * (10**quantisation_factor))) for i_c in range(3)] for i_v in range(len(vertices))]) # TODO : voir pour ajouter N//4

    bits = np.array([[powmod(quantized[i][j]//qim_step, 1, 2) for j in range(3)] for i in range(len(quantized))])
    base = (quantized // qim_step) * qim_step
    pre_marked = np.where(bits == 0, base + qim_step // 2, base - qim_step //2)

    for i_v,_  in enumerate(vertices):
        for i_c, _ in enumerate(vertices[i_v]):
            if (pre_marked[i_v][i_c]//qim_step)%2 == 1: 
                print(f"pre-marque = 1; {(pre_marked[i_v][i_c]//qim_step)%2}")

    # for i_v,_  in enumerate(vertices):
    #     for i_c, _ in enumerate(vertices[i_v]):
    #         pre_marked[i_v][i_c] = np.floor(vertices[i_v][i_c] * (10**quantisation_factor)).astype(int)
    #         if (pre_marked[i_v][i_c]//qim_step) % 2 == 0:
    #             pre_marked[i_v][i_c] = int(qim_step * (pre_marked[i_v][i_c]//qim_step) + (qim_step / 2))
    #         else:
    #             pre_marked[i_v][i_c] = int(qim_step * (pre_marked[i_v][i_c]//qim_step) - (qim_step / 2))

    #         pre_marked[i_v][i_c] = pre_marked[i_v][i_c] % encryption_keys["public"][0]

    #         if (pre_marked[i_v][i_c]//qim_step)%2 == 1: 
    #             print(f"pre-marque = 1; {(pre_marked[i_v][i_c]//qim_step)%2}")

    start_time_encryption = time.time()
    # encrypted_vertices = None
    encrypted_vertices = paillier.encrypt_vertices(pre_marked, encryption_keys["public"])
    time_encryption = time.time() - start_time_encryption

    return encrypted_vertices, pre_marked, {"time_encryption": time_encryption}


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
