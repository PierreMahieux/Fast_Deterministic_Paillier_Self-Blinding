import os
from gmpy2 import mpz, powmod, invert, gcd, mpz_random, random_state, mpz_urandomb, next_prime
import time
import datetime, hashlib
import numpy as np

from src.utils import paillier, util, mesh_utils

def embed(vertices: np.array, watermark: list, encryption_keys: dict, config: dict) -> dict:
    block_size = config["block_size"]
    num_message = config["num_message"]

    return None


def extract(vertices: np.array, encryption_keys: dict, quantisation_step: int, watermarks_sizes: tuple[int, int]) -> dict:

    return None

def preprocess(model: dict[np.array, np.array], config: dict) -> dict:
    k, alpha, encryption_keys = choose_block_size_and_parameters(config["block_size"])
    return None

def choose_block_size_and_parameters(b) -> tuple:
    """
    Étape 1: Choisir la taille de bloc et déduire les paramètres
    
    b: taille du bloc (nombre de vertices, doit être impair)
    
    Retourne: k, alpha, (pub_key, priv_key)
    """
    if b % 2 == 0:
        raise ValueError("La taille de bloc b doit être impaire")
    
    # Équation de l'article: 2k + 1 = 69b
    k = (69 * b - 1) // 2
    
    # alpha = k - 7 * nombre_axes * b = k - 7 * 3 * b = k - 21b
    alpha = k - 21 * b
    
    if alpha < 0:
        raise ValueError(f"Alpha doit être positif. Avec b={b}, alpha={alpha}. Augmentez b.")
    
    # La clé publique de Paillier doit avoir n = k + 1 bits
    key_size = k + 1
    
    # Générer les clés de Paillier
    encryption_keys = paillier.generate_keys(key_size)
    public_key = encryption_keys["public"]
    N_len  = public_key[0].bit_length()
    N_2_len= ((public_key[0])**2).bit_length()
    while (N_len!=k+1) and N_2_len!=2*k+1:
        encryption_keys = paillier.generate_keys(key_size)
    
    print(f"N_len= {N_len}") 
    print(f"N_2_len= {N_2_len}")    
    return k, alpha, encryption_keys