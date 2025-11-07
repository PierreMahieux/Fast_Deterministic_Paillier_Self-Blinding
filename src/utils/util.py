import hashlib
import os
import uuid
import datetime
import json
import ecdsa


def genereate_signing_keys() -> dict:
    sk = ecdsa.SigningKey.generate(curve=ecdsa.NIST256p)
    vk = sk.verifying_key
    return {"signing": sk, "verifying": vk}

def verify_signature(signature, message: bytes, verifying_key) -> bool:
    try:
        verifying_key.verify(signature, message)
        return True
    except ecdsa.BadSignatureError:
        return False

def generate_signature(message: bytes, signing_key) -> bytes:
    return signing_key.sign(message)

def hash_string_to_bits(message: str):
    assert(message!=None)
    
    hash_bytes = hashlib.sha256(message.encode('utf-8')).digest()
    
    return bytes_to_bits(hash_bytes)

def int_to_bits(value, num_bits):
    value = int(value)
    
    bits = []
    for i in range(num_bits):
        bits.append((value >> (num_bits - 1 - i)) & 1)
    return bits

def bits_to_int(bits):
    value = 0
    for i, bit in enumerate(reversed(bits)):
        value += int(bit) * (2 ** i)
    return int(value)

def bytes_to_bits(byte_message: bytes) -> list:
    bits = []
    for byte in byte_message:
        binary_string = format(byte, '08b') 
        for bit_char in binary_string:
            bits.append(int(bit_char))
    
    return bits

def bits_to_bytes(bits_message: list) -> bytes:
    s = ''.join(str(x) for x in bits_message)[0::]
    return int(s, 2).to_bytes(len(s) // 8, byteorder='big')

def write_report(results: dict) -> None:
    try:
        os.makedirs(os.path.dirname(results["config"]["result_folder"]), exist_ok=True)
        filename = os.path.join(results["config"]["result_folder"], "report.txt")

        results_json = json.dumps(results)

        with open(filename, 'w') as file:
            json.dump(results, file, indent=4, sort_keys=True)
            # file.write("{\n")
            # for key, value in results.items():
            #     file.write(f"\"{key.replace("'", "\"")}\": {value.replace("'", "\"")},\n")
            # file.write("\n}")
        
        print(f"Rapport sauvegardé dans {filename}")
    except Exception as e:
        print(f"Erreur lors de l'écriture du rapport': {e}")  

    return None

def compare_bits(original_bits, extracted_bits):
    min_len = min(len(original_bits), len(extracted_bits))
    
    if min_len == 0:
        print("Erreur: Au moins une séquence est vide")
        return 1
    
    errors = 0
    for i in range(min_len):
        if original_bits[i] != extracted_bits[i]:
            errors += 1
    
    ber = errors / min_len
    
    return ber