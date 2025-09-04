import os
from gmpy2 import mpz, powmod, invert, gcd, mpz_random, random_state, mpz_urandomb, next_prime
import time
import datetime, hashlib
import numpy as np

from src.utils import paillier, util, mesh_utils

def preprocess(model: dict[np.array, np.array], encryption_keys: dict, config: dict) -> dict:
    print("Pre-processing")
    vertices = model["vertices"]
    quantisation_factor = config["quantisation_factor"]
    qim_step = config["qim_step"]

    # Quantisation
    quant_vertices = np.floor(vertices * (10**quantisation_factor)).astype(int)

    # Reverse quantisation to compare quantified and recovered meshes
    save_mesh = []
    for v in quant_vertices:
        cs = []
        for c in v:
            cs.append(c/(10**quantisation_factor))
        save_mesh.append(cs)

    
    preprocessed_vertices = _encryption_preprocessing(quant_vertices, encryption_keys["public"][0])

    pre_watermarked_vertices = preprocessed_vertices.copy()
    for v in pre_watermarked_vertices:
        for i in range(3):
            if (v[i]//qim_step) % 2 == 0:
                v[i] = qim_step * (v[i]//qim_step) + (qim_step / 2)
            else:
                v[i] = qim_step * (v[i]//qim_step) - (qim_step / 2)

    start_time_encryption = time.time()
    enc_vertices = _encrypt_vertices(pre_watermarked_vertices, encryption_keys["public"])
    time_encryption = time.time() - start_time_encryption

    return {"pre_watermarked_vertices": pre_watermarked_vertices, "encrypted_vertices": enc_vertices, "time_encryption": time_encryption}

def embed(vertices: np.array, watermarks: dict, keys: dict, config: dict) -> dict:
    start_time_embedding = time.time()
    qim_step = config["qim_step"]
    qim_watermark = watermarks["qim"]
    encryption_keys = keys["encryption"]
    signing_keys = keys["signing"]

    public_encryption_key = encryption_keys["public"]
    start_embedding = time.time()

    embedded_vertices = _qim_embedding(vertices, qim_watermark, qim_step, public_encryption_key)
    time_qim = time.time() - start_embedding

    start_self_blinding = time.time()

    prepared_vertices = _embed_0_in_last_vertices(embedded_vertices, public_encryption_key, config["length_signature"])

    mesh_hash = hashlib.sha256(np.array2string(prepared_vertices).encode('utf-8')).digest()
    mesh_signature = util.generate_signature(mesh_hash, signing_keys["signing"])

    signed_vertices = _sign_mesh(prepared_vertices, mesh_signature, encryption_keys["public"][0])
    time_self_blinding = time.time() - start_self_blinding

    return {"signed_vertices": signed_vertices, "time_qim": time_qim, "time_self_blinding": time_self_blinding}

def extract(signed_vertices: np.array, keys: dict, quantisation_factor: int, watermarks_length: tuple, config: dict) -> dict:
    encryption_keys = keys["encryption"]
    signing_keys = keys["signing"]

    extracted_signature = _extract_signature(signed_vertices, encryption_keys["public"][0], watermarks_length[1])
    extracted_signature = util.bits_to_bytes(extracted_signature)

    unsigned_vertices = _embed_0_in_last_vertices(signed_vertices, encryption_keys["public"], watermarks_length[1])
    extracted_hash = hashlib.sha256(np.array2string(unsigned_vertices).encode('utf-8')).digest()

    model_is_signed = util.verify_signature(extracted_signature, extracted_hash, signing_keys["verifying"])

    extracted_watermark = None
    decrypted_vertices = None
    if model_is_signed:
        decrypted_vertices = _decrypt_vertices(unsigned_vertices, encryption_keys)
        extracted_watermark = _qim_extraction(decrypted_vertices, config["qim_step"])

    return {"model_signed": model_is_signed, "extracted_watermark": extracted_watermark, "extracted_signature": extracted_signature, "decrypted_vertices": decrypted_vertices}

def _extract_signature(vertices: np.array, N, signature_length) -> list:
    signature = []

    for i in range(-signature_length, 0):
        if 1 <= vertices[i//3][i%3] <= (N**2) // 2:
            signature.append(0)
        elif (N**2) // 2 + 1 <= vertices[i//3][i%3] <= (N**2) + 1:
            signature.append(1)

    return signature

def _sign_mesh(vertices: np.array, signature: bytes, N) -> np.array:
    print("Self-blinding embedding")
    signed_vertices = vertices.copy()
    signature_bits = util.bytes_to_bits(signature)

    for i in range(-len(signature_bits), 0):
        if signature_bits[i] == 1:
            signed_vertices[i//3][i%3] = -signed_vertices[i//3][i%3] % N**2

    return signed_vertices

def _embed_0_in_last_vertices(vertices: np.array, public_encryption_key: tuple, length_watermark: int) -> np.array:
    prepared_vertices = vertices.copy()
    N = public_encryption_key[0]

    for j in range(-length_watermark, 0):
        if (N**2)//2 + 1 <= prepared_vertices[j//3][j%3] <= N**2 - 1:
            prepared_vertices[j//3][j%3] = -prepared_vertices[j//3][j%3] % N**2

    return prepared_vertices

def _qim_extraction(vertices, qim_step) -> list:
    extract_w = []
    for i in range(len(qim_watermark)):
        extract_w.append((vertices[i//3][i%3]//qim_step) % 2)

    return extract_w


def _qim_embedding(vertices: np.array, watermark: list, qim_step: int, public_encryption_key: tuple) -> np.array:
    print("QIM embedding")
    embedded_vertices = vertices.copy()

    encrypted_step = paillier.encrypt_given_r(qim_step, public_encryption_key, 1)

    for i in range(len(watermark)):
        if watermark[i] == 1:
            embedded_vertices[i//3][i%3] = (embedded_vertices[i//3][i%3] * encrypted_step) % public_encryption_key[0]**2

    return embedded_vertices

def _encryption_preprocessing(vertices: np.array, N) -> np.array:
    # Forward mapping
    vertices_positive = []
    for vertex in vertices:
        positive_vertex = []
        for coord in vertex:
            if coord >= 0:
                positive_coord = coord
            else:
                positive_coord = coord + N
            positive_vertex.append(positive_coord)
        vertices_positive.append(positive_vertex)
    
    return np.array(vertices_positive)

def _encrypt_vertices(vertices, public_key) -> np.array:
    """
    vertices: ensemble des noeuds de l'objet initiale
    pub_key: clé publique de paillier
    k : paramètre de précision (nombre de décimales à conserver) pour le preprocessing
    
    Chiffre toutes les coordonnées des vertices.
    
    Retourne tous les noeuds chiffrés sous forme de tableau numpy et le clés de Paillier"""
    
    print("Encryption")
    #Chiffrement
    encrypted_vertices = []
    for vertex in vertices:
        encrypted_vertex = []
        for coord in vertex:
            enc_coord = paillier.encrypt(int(coord), public_key)
            encrypted_vertex.append(enc_coord)
        encrypted_vertices.append(encrypted_vertex)
    return np.array(encrypted_vertices)

def _decrypt_vertices(vertices, paillier_keys) -> np.array:
    """
    vertices : les noeuds chiffrés
    paillier_keys : clé publique et privée de Paillier
    
    Déchiffre toutes les coordonnées
    
    Retoune tous les noeuds déchiffrés en tableau numpy"""
    print("Decryption")
    #temps
    pub_key = paillier_keys["public"]
    priv_key = paillier_keys["secret"]
    decrypted_vertices = []
    
    for vertex in vertices:
        decrypted_vertex = []
        for coord in vertex:
            # dec_coord = paillier.decrypt_CRT(coord, priv_key, pub_key)
            dec_coord = paillier.decrypt(coord, priv_key, pub_key)
            decrypted_vertex.append(dec_coord)
        decrypted_vertices.append(decrypted_vertex)
    return np.array(decrypted_vertices)

def recover_mesh(vertices: np.array, public_key: tuple, quantisation_factor: int) -> np.array:
    print("Vertices recovery")
    recovered_vertices = []
    N = public_key[0]

    for vertex in vertices:
        restored_vertex = []
        for coord in vertex:
            if coord > N//2 : 
                restored_coord = (coord - N) # // 2
            else:  
                restored_coord = (coord) # // 2
            
            restored_vertex.append(restored_coord / (10**quantisation_step))
        recovered_vertices.append(restored_vertex)

    return np.array(recovered_vertices).astype(float)