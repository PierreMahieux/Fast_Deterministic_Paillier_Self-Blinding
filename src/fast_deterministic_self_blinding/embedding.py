import time
import hashlib
import numpy as np
import gmpy2

from src.utils import util, paillier

def embed(vertices: np.array, qim_watermark: tuple, keys: dict, config: dict) -> (np.array, dict):
    start_time_embedding = time.time()
    qim_step = config["qim_step"]
    encryption_keys = keys["encryption"]
    signing_keys = keys["signing"]

    public_encryption_key = encryption_keys["public"]
    start_embedding = time.time()
    embedded_vertices = vertices.copy()
    embedded_vertices = _qim_embedding(vertices, qim_watermark, qim_step, public_encryption_key)
    time_qim = time.time() - start_embedding
    
    start_self_blinding = time.time()

    prepared_vertices = _embed_0_in_first_vertices(embedded_vertices, public_encryption_key, config["signature_length"])

    mesh_hash = hashlib.sha256(np.array2string(prepared_vertices).encode('utf-8')).digest()
    mesh_signature = util.generate_signature(mesh_hash, signing_keys["signing"])

    signed_vertices = _sign_mesh(prepared_vertices, mesh_signature, encryption_keys["public"][0])
    time_self_blinding = time.time() - start_self_blinding
    
    return signed_vertices, {"time_qim": time_qim, "time_self_blinding": time_self_blinding}



def _qim_embedding(vertices: np.array, watermark: list, qim_step: int, public_encryption_key: tuple) -> np.array:
    print("QIM embedding")
    embedded_vertices = vertices.copy()

    encrypted_qim_step = paillier.encrypt_given_r(qim_step, public_encryption_key, 1)

    for i in range(len(watermark)):
        if watermark[i] == 1:
            embedded_vertices[i//3][i%3] = gmpy2.powmod(embedded_vertices[i//3][i%3] * encrypted_qim_step, 1, public_encryption_key[0]**2)

    return embedded_vertices

def _embed_0_in_last_vertices(vertices: np.array, public_encryption_key: tuple, length_watermark: int) -> np.array:
    prepared_vertices = vertices.copy()
    N = public_encryption_key[0]

    for j in range(-length_watermark, 0):
        if (N**2 + 1)//2 <= prepared_vertices[j//3][j%3] <= N**2 - 1:
            prepared_vertices[j//3][j%3] = (-prepared_vertices[j//3][j%3]) % N**2

    return prepared_vertices

def _embed_0_in_first_vertices(vertices: np.array, public_encryption_key: tuple, length_watermark: int) -> np.array:
    prepared_vertices = vertices.copy()
    N = public_encryption_key[0]

    for j in range(0, length_watermark):
        if (N**2 + 1)//2 <= prepared_vertices[j//3][j%3] <= N**2 - 1:
            prepared_vertices[j//3][j%3] = (-prepared_vertices[j//3][j%3]) % N**2

    return prepared_vertices

def _sign_mesh(vertices: np.array, signature: bytes, N) -> np.array:
    print("Self-blinding embedding")
    signed_vertices = vertices.copy()
    signature_bits = util.bytes_to_bits(signature)

    for i in range(0, len(signature_bits)):
        if signature_bits[i] == 1:
            signed_vertices[i//3][i%3] = (-signed_vertices[i//3][i%3]) % N**2

    return signed_vertices
