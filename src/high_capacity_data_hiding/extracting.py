import numpy as np

from src.utils import paillier, util

def extract_messages(encrypted_blocks_data):
    """
    Extrait tous les messages et les reconstruit en messages de 256 bits
    
    encrypted_blocks_data: dictionnaire contenant les blocs chiffrés tatoués
    
    Retourne: liste de messages de 256 bits
    """
    alpha = encrypted_blocks_data['parameters']['alpha']
    
    # Extraire tous les segments
    all_segments = _extract_all_messages(encrypted_blocks_data)
    
    # Reconstruire chaque message
    reconstructed_messages = []
    
    print("\n=== RECONSTRUCTION DES MESSAGES ===")
    for i, segments in enumerate(all_segments):
        message = _reconstruct_message_from_segments(segments, alpha)
        reconstructed_messages.append(message)
    
    return reconstructed_messages

def _extract_all_messages(encrypted_blocks_data):
    """
    Extrait tous les messages tatoués
    
    encrypted_blocks_data: dictionnaire contenant les blocs chiffrés tatoués
    
    Retourne: liste de messages (chaque message est une liste de segments)
    """
    print("\n=== EXTRACTION DES MESSAGES ===")
    
    messages = []
    current_idx = 0
    total_blocks = len(encrypted_blocks_data['encrypted_blocks'])
    
    while current_idx < total_blocks:
        # Vérifier si on a un bloc avec flag=1 (début de message)
        block_info = encrypted_blocks_data['encrypted_blocks'][current_idx]
        
        if 'watermarked_flagged_encrypted_block' in block_info:
            block_to_check = block_info['watermarked_flagged_encrypted_block']
        else:
            block_to_check = block_info['encrypted_block']
        
        if _check_block_flag(block_to_check) == 1:
            # Extraire le message
            segments, next_idx = _extract_single_message(encrypted_blocks_data, current_idx)
            
            if segments:
                messages.append(segments)
                print(f"Message {len(messages)} extrait: {len(segments)} segments")
                print(f"  Blocs B{current_idx+1} à B{current_idx+len(segments)}")
            
            current_idx = next_idx
        else:
            # Bloc avec flag=0, passer au suivant
            current_idx += 1
    
    print(f"\nTotal: {len(messages)} messages extraits")
    return messages

def _extract_single_message(encrypted_blocks_data, start_idx, skip_decryption=False):
    """
    Extrait un message à partir d'un index donné
    
    encrypted_blocks_data: dictionnaire contenant les blocs chiffrés tatoués
    start_idx: index du premier bloc du message
    skip_decryption: si True, simule l'extraction sans déchiffrer (pour les tests)
    
    Retourne: (segments extraits, index du prochain bloc après le message)
    """
    blocks = encrypted_blocks_data['encrypted_blocks']
    k = encrypted_blocks_data['parameters']['k']
    alpha = encrypted_blocks_data['parameters']['alpha']
    pub_key = encrypted_blocks_data['encryption_keys']["public"]
    priv_key = encrypted_blocks_data['encryption_keys']["secret"]
    
    segments = []
    current_idx = start_idx
    
    # Extraire les segments tant qu'on trouve des blocs avec flag=1
    while current_idx < len(blocks):
        block_info = blocks[current_idx]
        
        # Utiliser le bloc tatoué s'il existe
        if 'watermarked_flagged_encrypted_block' in block_info:
            block_to_check = block_info['watermarked_flagged_encrypted_block']
        else:
            block_to_check = block_info['encrypted_block']
        
        # Vérifier le flag
        if _check_block_flag(block_to_check) == 1:
            # Flag=1, c'est un bloc du message
            
            if skip_decryption:
                # Pour les tests : créer un segment simulé
                segment = [0] * alpha
                segments.append(segment)
            else:
                # Le bloc est un entier, pas des bits
                # Déchiffrer le bloc
                decrypted = _decrypt_block(block_to_check, k, priv_key, pub_key)
                
                # Extraire le segment
                segment = _extract_segment_from_decrypted_block(decrypted, k, alpha)
                segments.append(segment)
            
            current_idx += 1
        else:
            # Flag=0, fin du message
            break
    
    # Retourner les segments et l'index du prochain bloc disponible
    # (sauter le bloc de séparation avec flag=0)
    next_idx = current_idx + 1 if current_idx < len(blocks) else current_idx
    
    return segments, next_idx

def _extract_segment_from_decrypted_block(decrypted_block_bits, k, alpha):
    """
    Extrait le segment de message des alpha derniers bits des k premiers bits
    
    decrypted_block_bits: bloc déchiffré avec padding (2k+1 bits)
    k: nombre de bits utiles
    alpha: taille du segment
    
    Retourne: segment de message (alpha bits)
    """
    # Prendre seulement les k premiers bits (sans le padding)
    k_bits = decrypted_block_bits[:k]
    
    # Extraire les alpha derniers bits de ces k bits
    segment_start = k - alpha
    segment = k_bits[segment_start:k]
    
    return segment

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

def _check_block_flag(block):
    """
    Vérifie le flag d'un bloc (parité)
    
    block: bloc en bits ou entier
    
    Retourne: 0 si pair, 1 si impair
    """
    if isinstance(block, (int, type(block))):  # Si c'est déjà un entier
        block_int = block
    else:  # Si c'est une liste de bits
        block_int = util.bits_to_int(block)
    
    return block_int % 2


def _reconstruct_message_from_segments(segments, alpha, expected_size=256):
    """
    Reconstruit un message de 256 bits à partir de ses segments
    
    segments: liste de segments (chaque segment a alpha bits)
    alpha: taille d'un segment
    expected_size: taille attendue du message (256 bits par défaut)
    
    Retourne: message de 256 bits
    """
    # Concaténer tous les segments
    all_bits = []
    for segment in segments:
        all_bits.extend(segment)
    
    # Le dernier segment peut avoir du padding
    # Calculer combien de bits utiles on a
    num_segments = len(segments)
    total_bits_without_padding = expected_size
    
    # Si on a plus de bits que nécessaire, c'est qu'il y a du padding
    if len(all_bits) > expected_size:
        # Prendre seulement les 256 premiers bits
        message_bits = all_bits[:expected_size]
    else:
        message_bits = all_bits
    
    return message_bits
