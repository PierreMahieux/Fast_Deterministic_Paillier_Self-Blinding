import os
from gmpy2 import mpz, powmod, invert, gcd, mpz_random, random_state, mpz_urandomb, next_prime
import time
import datetime, hashlib
import numpy as np

from src.high_capacity_data_hiding.preprocessing import preprocess
from src.high_capacity_data_hiding.embedding import embed
from src.high_capacity_data_hiding.extracting import extract_messages
from src.utils import paillier, util, mesh_utils

def run(config: dict, encryption_keys: dict, watermarks: tuple, model):
    print("Run HCDH")
    vertices = model["vertices"]
    faces = model["faces"]
    result = {"config": config}

    result_preprocess = preprocessing(model, config)
    params = result_preprocess["parameters"]
    params["key_size"] = result_preprocess["encryption_keys"]["public"][0].bit_length()
    result = result | result_preprocess

    start_encryption = time.time()
    encrypted_blocks_data = encrypt_all_blocks(result_preprocess)
    result["time_encryption"] = time.time() - start_encryption

    start_embedding = time.time()
    alpha = params['alpha']
    watermarked_data = embedding(encrypted_blocks_data, watermarks, alpha)
    result["time_embedding"] = time.time() - start_embedding

    start_extraction = time.time()
    extracted_messages = extracting(watermarked_data)
    result["time_extraction"] = time.time() - start_extraction

    decrypted_vertices = reconstruct_decrypt_vertices(encrypted_blocks_data)
    mesh_utils.save_3d_model(decrypted_vertices, model["faces"], os.path.join(config["result_folder"], "restored_decrypted.obj"))

    for i in range(config["number_messages"]):
        result[f"ber_{i}"] = util.compare_bits(watermarks[i], extracted_messages[i])
        print(result[f"ber_{i}"])

    return result

def preprocessing(model: dict[np.array, np.array], config: dict) -> dict:
    return preprocess(model, config)


def embedding(encrypted_blocks_data, messages_bits, alpha):
    return embed(encrypted_blocks_data, messages_bits, alpha)

def extracting(encrypted_blocks_data):
    return extract_messages(encrypted_blocks_data)

def reconstruct_decrypt_vertices(encrypted_blocks_data):
    """
    Reconstruit tous les vertices déchiffrés (avec tatouage)
    
    encrypted_blocks_data: dictionnaire contenant les blocs chiffrés tatoués
    
    Retourne: vertices déchiffrés tatoués
    """
    blocks = encrypted_blocks_data['encrypted_blocks']
    k = encrypted_blocks_data['parameters']['k']
    alpha = encrypted_blocks_data['parameters']['alpha']
    b = encrypted_blocks_data['parameters']['b']
    pub_key = encrypted_blocks_data['encryption_keys']["public"]
    priv_key = encrypted_blocks_data['encryption_keys']["secret"]
    n_vertices_original = encrypted_blocks_data['n_vertices_original']
    
    all_vertices = []
    
    print("\n=== RECONSTRUCTION DES VERTICES DÉCHIFFRÉS ===")
    
    for i, block_info in enumerate(blocks):
        # Utiliser le bloc tatoué s'il existe
        if 'watermarked_flagged_encrypted_block' in block_info:
            block_to_decrypt = block_info['watermarked_flagged_encrypted_block']
        else:
            block_to_decrypt = block_info['encrypted_block']
        
        # Reconstruire les vertices du bloc
        block_vertices = _reconstruct_decrypted_vertices_from_block(
            block_to_decrypt,
            block_info['metadata'],
            k, alpha, b,
            priv_key, pub_key
        )
        
        all_vertices.extend(block_vertices)
    
    # Enlever le padding
    decrypted_vertices = np.array(all_vertices[:n_vertices_original])
    
    print(f"Vertices reconstruits: {decrypted_vertices.shape}")
    return decrypted_vertices

def _reconstruct_decrypted_vertices_from_block(encrypted_block_bits, metadata, k, alpha, b, priv_key, pub_key):
    """
    Reconstruit les vertices déchiffrés d'un bloc
    
    encrypted_block_bits: bloc chiffré tatoué
    metadata: métadonnées pour la reconstruction
    k, alpha, b: paramètres
    priv_key, pub_key: clés de déchiffrement
    
    Retourne: vertices déchiffrés du bloc
    """
    # Déchiffrer le bloc
    decrypted_with_padding = _decrypt_block(encrypted_block_bits, k, priv_key, pub_key)
    
    # Reconstruire les vertices à partir du bloc déchiffré avec padding
    vertices = _reconstruct_vertices_from_block(decrypted_with_padding, metadata, b)
    
    return vertices

def _decrypt_block(encrypted_block, k, priv_key, pub_key):
    """
    Déchiffre un bloc et ajoute le padding nécessaire
    
    encrypted_block: bloc chiffré
    k: nombre de bits chiffrés
    priv_key: clé privée Paillier
    pub_key: clé publique Paillier
    
    Retourne: bloc déchiffré de 2k+1 bits avec padding
    """
    # Déchiffrer
    decrypted_int = paillier.decrypt_CRT(encrypted_block, priv_key, pub_key)
    
    # Convertir en k bits
    decrypted_bits = util.int_to_bits(decrypted_int, k)
    
    # Ajouter le padding (k+1 bits à 0)
    padding = [0] * ((2*k + 1) - len(decrypted_bits))
    decrypted_block_with_padding = np.concatenate([decrypted_bits, padding])
    
    return decrypted_block_with_padding

def _reconstruct_vertices_from_block(block, metadata, b):
    """
    Reconstruit les vertices à partir d'un bloc et des métadonnées
    
    block: liste de bits de taille 69b
    metadata: liste des (signe, exposant) pour chaque coordonnée
    b: nombre de vertices dans le bloc
    
    Retourne: liste de vertices reconstruits
    """
    # Vérifier les tailles
    assert len(block) == 69 * b, f"Block size should be {69*b}, got {len(block)}"
    assert len(metadata) == 3 * b, f"Metadata size should be {3*b}, got {len(metadata)}"
    
    # Initialiser un array numpy vide pour les mantisses
    mantisses = np.zeros((3 * b, 23), dtype=int)
    
    # Pour chaque position de bit
    for bit_position in range(23):
        # Pour chaque vertex
        for vertex_idx in range(b):
            # Pour chaque coordonnée (x, y, z)
            for coord_idx in range(3):
                mantisse_idx = vertex_idx * 3 + coord_idx
                block_idx = bit_position * 3 * b + vertex_idx * 3 + coord_idx
                bit = block[block_idx]
                mantisses[mantisse_idx, bit_position] = bit
    
    # Reconstruire les vertices
    vertices = []
    for vertex_idx in range(b):
        vertex = []
        for coord_idx in range(3):
            mantisse_idx = vertex_idx * 3 + coord_idx
            mantisse_bits = mantisses[mantisse_idx]
            signe, exposant = metadata[mantisse_idx]
            
            # Reconstruire la coordonnée
            coord = _reconstruct_coordinate_from_mantissa(mantisse_bits, signe, exposant)
            vertex.append(coord)
        vertices.append(vertex)
    
    return np.array(vertices)

def _reconstruct_coordinate_from_mantissa(mantisse_bits, signe, exposant):
    """
    Reconstruit une coordonnée à partir de sa mantisse, signe et exposant
    
    mantisse_bits: liste de 23 bits de mantisse
    signe: bit de signe (0 ou 1)
    exposant: valeur de l'exposant (8 bits)
    
    Retourne: coordonnée flottante reconstruite
    """
    # Convertir la mantisse en entier
    mantisse_int = util.bits_to_int(mantisse_bits)
    
    # Reconstruire la représentation IEEE 754
    # Signe (bit 31) | Exposant (bits 23-30) | Mantisse (bits 0-22)
    int_repr = (signe << 31) | (exposant << 23) | (mantisse_int & 0x7FFFFF)
    
    # Convertir en float
    coord = np.uint32(int_repr).view(np.float32)
    
    return coord




def encrypt_all_blocks(preprocessing_result):
    """
    Chiffre tous les blocs du modèle prétraité
    
    preprocessing_result: résultat du preprocessing
    
    Retourne: dictionnaire contenant les blocs chiffrés et toutes les métadonnées
    """
    blocks_data = preprocessing_result['blocks_data']
    k = preprocessing_result['parameters']['k']
    pub_key = preprocessing_result['encryption_keys']["public"]
    encrypted_blocks_list = []
    print(f"Taille effective de N = {pub_key[0].bit_length()}")
    print("\n=== CHIFFREMENT DES BLOCS ===")
    for block_info in blocks_data:
        watermarkable_block = block_info['watermarkable_block']
        
        # Chiffrer le bloc
        encrypted = _encrypt_block(watermarkable_block, k, pub_key)
        
        
        encrypted_blocks_list.append({
            'block_id': block_info['block_id'],
            'encrypted_block': encrypted,
            'metadata': block_info['metadata'],  # Pour la reconstruction
            'vertex_indices': block_info['vertex_indices']
        })
        
    print(f"Nombre de blocs chiffrés: {len(encrypted_blocks_list)}")
    
    # Retourner un dictionnaire avec toutes les informations nécessaires
    return {
        'encrypted_blocks': encrypted_blocks_list,
        'parameters': preprocessing_result['parameters'],
        'encryption_keys': preprocessing_result['encryption_keys'],
        'n_vertices_original': preprocessing_result['n_vertices_original']
    }

def _encrypt_block(watermarkable_block, k, pub_key):
    """
    Chiffre un bloc tatouable en ne chiffrant que les k premiers bits
    
    watermarkable_block: bloc de 2k+1 bits (69b bits)
    k: nombre de bits à chiffrer
    pub_key: clé publique Paillier
    
    Retourne: chiffré de taille théorique 2k+1 bits
    """
    # Extraire seulement les k premiers bits (MSB)
    k_bits = watermarkable_block[:k]
    
    # Convertir en entier
    k_int = util.bits_to_int(k_bits)
    
    # Chiffrer avec Paillier
    encrypted = paillier.encrypt(k_int, pub_key)
    
    return encrypted
