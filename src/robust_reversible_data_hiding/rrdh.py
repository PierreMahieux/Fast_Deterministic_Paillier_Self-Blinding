import os
from gmpy2 import mpz, powmod, invert, gcd, mpz_random, random_state, mpz_urandomb, next_prime
import time
import datetime, hashlib
import numpy as np

from src.utils import paillier, util, mesh_utils

def embed(model: dict[np.array, np.array], watermark: tuple[list, list[int]], encryption_keys: dict, config: dict) -> dict:

    return None


def extract(vertices: np.array, encryption_keys: dict, quantisation_step: int, watermarks_sizes: tuple[int, int]) -> dict:

    return None