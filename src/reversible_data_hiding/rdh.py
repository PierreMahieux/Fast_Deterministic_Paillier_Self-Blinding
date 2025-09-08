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

    # Quantization
    quant_vertices = np.floor(vertices * (10**quantisation_factor)).astype(int)

    # Reverse quantisation to compare quantified and recovered meshes
    save_mesh = []
    for v in quant_vertices:
        cs = []
        for c in v:
            cs.append(c/(10**quantisation_factor))
        save_mesh.append(cs)

    start_time_encryption = time.time()
    pre_pro_vertices = _encryption_preprocessing(quant_vertices, encryption_keys["public"][0])
    enc_vertices = _encrypt_vertices(pre_pro_vertices, encryption_keys["public"])
    time_encryption = time.time() - start_time_encryption
    
    return {"encrypted_vertices": enc_vertices, "time_encryption": time_encryption, "quantified_model": save_mesh}

def embed(vertices: np.array, watermark: tuple, public_encryption_keys: dict, config: dict) -> dict:
    start_time_embedding = time.time()
    embedded_vertices = _histogram_shifting_embedding(vertices, watermark[0], public_encryption_keys)

    start_self_blinding = time.time()
    embedded_vertices = _self_blinding_embedding(embedded_vertices, watermark[1], public_encryption_keys)
    time_self_blinding = time.time() - start_self_blinding

    time_embedding = time.time() - start_time_embedding

    return {"embedded_encrypted_model": embedded_vertices, "time_embedding": time_embedding, "time_self_blinding": time_self_blinding}

def _self_blinding_embedding(vertices, watermark, pub_key) -> np.array:
    #Version simple, on tatoue sur les premiers noeuds. Aprés, faut tatouer sur des coordonnées qu'on choisi. Ces coordonnées seront en paramétres. 
    #Remarques : pas robuste aux attques de bruits.
    """
    vertices_after_first_tier : les noeuds aprés le premier tatouage
    management_bits : ensemble des bits à tatouer 
    pub_key : clé publique de Paillier
    
    Deuxième tatouage pour la gestion cloud. 
    Version simple : Les bits sont tatoués sur les premiers noeuds des vertices_after_first_tier jusqu'à ce que les bits à tatouer se terminent. 
    Si les bits à tatouer sont terminés, les autres noeuds ne sont pas modifiés. 
    
    Retourne tous les noeuds (tatoués et non tatoués) obtenue aprés le deuxième tatouage (type list).
    """
    print("Self-blinding embedding")
    #print(f"Bits à tatouer : {management_bits}")
    #print(f"Nombre de bits à tatouer : {len(management_bits)}")
    
    N, _ = pub_key
    N2 = N ** 2
    marked_vertices = []
    bit_index = 0
    
    for vertex in vertices:
        cloud_vertex = []
        for coord in vertex:
            if bit_index < len(watermark):
                bit = watermark[bit_index]
                
                # Vérifier la parité actuelle
                current_parity = coord % 2
                
                if current_parity != bit:
                    # Modifier la parité via self-blinding
                    P = paillier.generate_r(N)
                    ## Chercher un P tel que la nouvelle parité soit correcte. !!!! Peut être trés long !!!!!!
                    while True:
                        P = paillier.generate_r(N)
                        P_N = powmod(P, N, N2)
                        new_coord = (coord * P_N) % N2
                        if new_coord % 2 == bit:
                            coord = new_coord
                            break
                
                cloud_vertex.append(coord)
                bit_index += 1
                
            else: cloud_vertex.append(coord)
        
        marked_vertices.append(cloud_vertex)
        
    return marked_vertices

def _histogram_shifting_embedding(vertices, watermark, public_key) -> tuple[list, list, np.array]:
    print("Histogram shifting embedding")
    N = public_key[0]
    N2 = N ** 2
    marked_vertices = []
    bit_index = 0

    for vertex in vertices:
        marked_vertex = []
        for coord in vertex:
            # Expansion : élévation au carré
            expanded = powmod(coord, 2, N2)
            if bit_index < len(watermark):
                
                # Chiffrement du bit
                bit = watermark[bit_index]
                enc_bit = paillier.encrypt(bit, public_key)
                
                # Multiplication (addition en clair)
                marked_coord = (expanded * enc_bit) % N2
                marked_vertex.append(marked_coord)
                
                bit_index += 1
                
            else : marked_vertex.append(expanded)
                                 
        marked_vertices.append(marked_vertex)

    return marked_vertices

def extract(vertices: np.array, encryption_keys: dict, quantisation_factor: int, watermarks_sizes: tuple) -> dict:
    start_time = time.time()
    self_blinding_watermark = _self_blinding_extraction(vertices, watermarks_sizes[1])
    time_extraction_self_blinding = time.time() - start_time

    start_time = time.time()
    decrypted_vertices = _decrypt_vertices(vertices, encryption_keys)
    time_decryption = time.time() - start_time

    start_time = time.time()
    histogram_shifting_watermark = _histogram_shifting_extraction(decrypted_vertices, watermarks_sizes[0], encryption_keys["public"])
    time_extraction_histogram_shifting = time.time() - start_time
    
    return {"histogram_shifting_watermark": histogram_shifting_watermark, "self_blinding_watermark": self_blinding_watermark, "decrypted_vertices": decrypted_vertices, "time_extraction_histogram_shifting": time_extraction_histogram_shifting, "time_extraction_self_blinding": time_extraction_self_blinding, "time_decryption": time_decryption}

def _histogram_shifting_extraction(vertices, watermark_size, public_key) -> list:
    
    print("Histogram shifting extraction")
    extracted_bits = []
    extracted_bit_count = 0
    N = public_key[0]  

    for vertex in vertices:
        for coord in vertex:
            if extracted_bit_count < watermark_size:
                if coord > N//2 : 
                    bit = 1 - int(coord % 2)
                else:  
                    bit = int(coord % 2)
                extracted_bits.append(bit)
                extracted_bit_count += 1
    
    return extracted_bits


def _self_blinding_extraction(vertices, watermark_size) -> list:
    """
    vertices_after_second_tier: les noeuds (tatoués et non tatoués) obtenue aprés le deuxième tatouage
    management_bits : ensemble des bits déjà tatoué
    
    Extraction des bits du cloud (dans le domaine chiffré)
    
    Retourne les bits qui ont été tatoués lors du deuxième tatouage (type list) .
    """
    print("Self-blinding extraction")
    
    extracted_watermark = []
    bit_count = 0
    
    for vertex in vertices:
        for coord in vertex:
            if bit_count < watermark_size:
                # Extraire la parité
                bit = int(coord % 2)
                extracted_watermark.append(bit)
                bit_count += 1
            else: break
        if bit_count >= watermark_size: break
    
    ##print(f"Bits du deuxième tatouage extraits: {extracted_bits}")
    #print(f"Nombre de bits du deuxième tatouage: {len(extracted_bits)}")
    # print( f"\nExtraction deuxième tatouage terminé !!!!!!!!!!!!!!!!!" if extracted_bits==management_bits else f"Erreur extraction")
    return extracted_watermark

def recover_mesh(vertices: np.array, public_key: tuple, quantisation_factor: int) -> np.array:
    print("Vertices recovery")
    recovered_vertices = []
    N = public_key[0]

    for vertex in vertices:
        restored_vertex = []
        for coord in vertex:
            if coord > N//2 : 
                restored_coord = (coord - N) // 2
            else:  
                restored_coord = (coord) // 2
            
            restored_vertex.append(restored_coord / (10**quantisation_factor))
        recovered_vertices.append(restored_vertex)

    return np.array(recovered_vertices).astype(float)

def _decrypt_vertices(vertices_after_second_tier, paillier_keys) -> np.array:
    """
    vertices_after_second_tier : les noeuds obtenus aprés le deuxième tatouage ou ceux obtenus aprés premier tatouage
    paillier_keys : clé publique et privée de Paillier
    
    Déchiffre toutes les coordonnées
    
    Retoune tous les noeuds déchiffrés en tableau numpy"""
    print("Decryption")
    #temps
    pub_key = paillier_keys["public"]
    priv_key = paillier_keys["secret"]
    decrypted_vertices = []
    
    for vertex in vertices_after_second_tier:
        decrypted_vertex = []
        for coord in vertex:
            dec_coord = paillier.decrypt_CRT(coord, priv_key, pub_key)
            decrypted_vertex.append(dec_coord)
        decrypted_vertices.append(decrypted_vertex)
    return np.array(decrypted_vertices)

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