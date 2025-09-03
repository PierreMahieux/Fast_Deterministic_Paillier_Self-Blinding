import os
from gmpy2 import mpz, powmod, invert, gcd, mpz_random, random_state, mpz_urandomb, next_prime
import time
import datetime, hashlib
import numpy as np

from src.utils import paillier, util, mesh_utils

def preprocess(model: dict[np.array, np.array], encryption_keys: dict, config: dict) -> dict:
    print("Pre-processing")
    vertices = model["vertices"]
    quantisation_delta = config["quantisation_delta"]
    qim_step = config["qim_step"]

    # Quantisation
    quant_vertices = np.floor(vertices * (10**quantisation_delta)).astype(int)

    # Reverse quantisation to compare quantified and recovered meshes
    save_mesh = []
    for v in quant_vertices:
        cs = []
        for c in v:
            cs.append(c/(10**quantisation_delta))
        save_mesh.append(cs)

    
    preprocessed_vertices = _encryption_preprocessing(quant_vertices, encryption_keys["public"][0])
    
    assert(np.all(np.greater(preprocessed_vertices, np.zeros((len(preprocessed_vertices), 3)))))

    pre_watermarked_vertices = []
    for v in preprocessed_vertices:
        v_0 = []
        for i in range(3):
            if (v[i]//qim_step) % 2 == 0:
                v_0.append(qim_step * (v[i]//qim_step) + (qim_step / 2))
            else:
                v_0.append(qim_step * (v[i]//qim_step) - (qim_step / 2))
        pre_watermarked_vertices.append(v_0)

    start_time_encryption = time.time()
    enc_vertices = _encrypt_vertices(pre_watermarked_vertices, encryption_keys["public"])
    time_encryption = time.time() - start_time_encryption

    return {"pre_watermarked_vertices": pre_watermarked_vertices, "encrypted_vertices": enc_vertices, "time_encryption": time_encryption}

def embed(vertices: np.array, watermarks: dict, encryption_keys: dict, config: dict) -> dict:
    start_time_embedding = time.time()
    qim_step = config["qim_step"]
    qim_watermark = watermarks["qim"]
    public_encryption_key = encryption_keys["public"]
    start_embedding = time.time()

    embedded_vertices = _qim_embedding(vertices, qim_watermark, qim_step, public_encryption_key)
    time_qim = time.time() - start_embedding

    ## TODO : Vérifier bon fonctionnement QIM
    dec_vec = _decrypt_vertices(embedded_vertices, encryption_keys)
    extract_w = []
    for i in range(len(qim_watermark)):
        extract_w.append((dec_vec[i//3][i%3]//qim_step) % 2)

    assert(extract_w == watermarks["qim"])

    start_self_blinding = time.time()
    prepared_vertices = _preprocess_deterministic_self_blinding(embedded_vertices, public_encryption_key, len(watermarks["self_blinding"]))
    return None

def _preprocess_deterministic_self_blinding(vertices: np.array, public_encryption_key: tuple, length_watermark: int) -> np.array:
    prepared_vertices = vertices.copy()
    N = public_encryption_key[0]

    for j in range(-length_watermark, 0):
        c_j = prepared_vertices[j]
        for k in range(3):
            c_j_k = []
            if (N**2)//2 + 1 <= c_j[k] <= N**2 + 1 :
                c_j_k.append(-c_j[k] % N**2)
        prepared_vertices = np.append(prepared_vertices, c_j_k)
    return prepared_vertices

def _qim_embedding(vertices: np.array, watermark: list, qim_step: int, public_encryption_key: tuple) -> np.array:
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

def recover_mesh(vertices: np.array, public_key: tuple, quantisation_delta: int) -> np.array:
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
            
            restored_vertex.append(restored_coord) # / (10**quantisation_step))
        recovered_vertices.append(restored_vertex)

    return np.array(recovered_vertices).astype(float)