import time
import numpy as np
import hashlib

from src.utils import paillier, util
from src.fast_deterministic_self_blinding.embedding import _embed_0_in_last_vertices

def extract(vertices: np.array, keys: dict, quantisation_factor: int, watermark_length: int, signature_length: int, qim_step: int) -> (np.array, dict):
    encryption_keys = keys["encryption"]
    signing_keys = keys["signing"]

    extracted_signature = _extract_signature(vertices, encryption_keys["public"][0], signature_length)
    extracted_signature = util.bits_to_bytes(extracted_signature)

    unsigned_vertices = _embed_0_in_last_vertices(vertices, encryption_keys["public"], signature_length)
    extracted_hash = hashlib.sha256(np.array2string(unsigned_vertices).encode('utf-8')).digest()

    model_is_signed = util.verify_signature(extracted_signature, extracted_hash, signing_keys["verifying"])

    extracted_watermark = None
    decrypted_vertices = None
    if model_is_signed:
        decrypted_vertices = paillier.decrypt_vertices(unsigned_vertices, encryption_keys)
        extracted_watermark = _qim_extraction(decrypted_vertices, qim_step, watermark_length)

    return decrypted_vertices, {"model_is_signed": model_is_signed, "extracted_watermark": extracted_watermark, "extracted_signature": extracted_signature}

def _extract_signature(vertices: np.array, N, signature_length) -> list:
    signature = []

    for i in range(-signature_length, 0):
        if 0 <= vertices[i//3][i%3] <= (N**2 - 1) // 2:
            signature.append(0)
        else :
            signature.append(1)

    return signature

def _qim_extraction(vertices, qim_step, watermark_length) -> list:
    extract_w = []
    for i in range(watermark_length):
        extract_w.append(int((vertices[i//3][i%3]//qim_step) % 2))

    return extract_w
