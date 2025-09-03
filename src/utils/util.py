import hashlib
import os
import uuid
import datetime
import json

def generate_watermark_bits(message):
    """
    Génère 256 bits de tatouage à partir d'un hash SHA-256
    
    message: le message à tatouer (str). Si None, génère un message aléatoire
    
    Retourne le hash du message en 256 bits 
    """
    assert(message!=None)
    
    # Calculer le SHA-256
    hash_bytes = hashlib.sha256(message.encode('utf-8')).digest()
    
    # Convertir en bits
    bits = []
    for byte in hash_bytes:
        binary_string = format(byte, '08b') 
        for bit_char in binary_string:
            bits.append(int(bit_char))
    
    print(f"Message: {message}")
    print(f"SHA-256: {hashlib.sha256(message.encode('utf-8')).hexdigest()}")
    
    return bits

def write_report(results: dict) -> None:
    try:
        os.makedirs(os.path.dirname(results["config"]["result_folder"]), exist_ok=True)
        filename = os.path.join(results["config"]["result_folder"], "report.txt")
        with open(filename, 'w') as file:
            file.write("{\n")
            file.write(f"\t\"config\": {json.dumps(results["config"])},\n")
            
            file.write(f"\t\"time_encryption\": {results["time_encryption"]},\n")
            file.write(f"\t\"time_embedding\": {results["time_embedding"]},\n")
            file.write(f"\t\"time_decryption\": {results["time_decryption"]},\n")
            file.write(f"\t\"time_extraction_histogram_shifting\": {results['time_extraction_histogram_shifting']},\n")
            file.write(f"\t\"time_extraction_self_blinding\": {results['time_extraction_self_blinding']},\n")
            file.write(f"\t\"BER_histogram_shifting\": {results['BER_histogram_shifting']},\n")
            file.write(f"\t\"BER_self_blinding\": {results['BER_self_blinding']}\n}}")
        
        print(f"Rapport sauvegardé dans {filename}")
    except Exception as e:
        print(f"Erreur lors de l'écriture du rapport': {e}")  

    return None

def compare_bits(original_bits, extracted_bits):
    """Compare deux séquences de bits et affiche les statistiques et retourne le BER [0,1]"""
    min_len = min(len(original_bits), len(extracted_bits))
    
    if min_len == 0:
        print("Erreur: Au moins une séquence est vide")
        return 1
    
    # Comparer bit par bit
    errors = 0
    for i in range(min_len):
        if original_bits[i] != extracted_bits[i]:
            errors += 1
    
    # Calculer le BER (Bit Error Rate)
    ber = errors / min_len
    
    return ber