from gmpy2 import powmod

from src.utils import paillier, util

def embed(encrypted_blocks_data, messages_bits, alpha):
    """
    Tatoue plusieurs messages dans les blocs
    
    encrypted_blocks_data: dictionnaire contenant les blocs chiffrés
    messages_bits: liste de messages en bits (chaque message est une liste de 256 bits)
    alpha: taille d'un segment en bits
    
    Retourne: dictionnaire mis à jour avec tous les messages tatoués
    """
    # Initialiser tous les blocs avec flag=0
    encrypted_blocks_data = _initialize_blocks_with_flags(encrypted_blocks_data)
    
    print(f"\n=== TATOUAGE DE {len(messages_bits)} MESSAGES ===")
    
    current_block_idx = 0
    embedding_info = []
    
    for msg_idx, message_bits in enumerate(messages_bits):
        print(f"\nMessage {msg_idx + 1}:")
        
        # Diviser le message en segments
        segments = _segment_message(message_bits, alpha)
        print(f"  - {len(segments)} segments de {alpha} bits")
        print(f"  - Blocs utilisés: B{current_block_idx + 1} à B{current_block_idx + len(segments)}")
        
        # Enregistrer les infos
        info = {
            'message_id': msg_idx,
            'start_block': current_block_idx,
            'num_segments': len(segments),
            'block_indices': list(range(current_block_idx, current_block_idx + len(segments)))
        }
        embedding_info.append(info)
        
        # Tatouer le message ET RÉCUPÉRER LE NOUVEL INDEX
        encrypted_blocks_data, current_block_idx = _embed_single_message(
            encrypted_blocks_data, 
            segments, 
            current_block_idx
        )
        
        # Le bloc de séparation a déjà un flag=0 (pair)
        if current_block_idx < len(encrypted_blocks_data['encrypted_blocks']):
            print(f"  - Bloc de séparation: B{current_block_idx} (flag=0)")
    
    # Ajouter les informations d'embedding
    encrypted_blocks_data['embedding_info'] = embedding_info
        
    return encrypted_blocks_data

def _initialize_blocks_with_flags(encrypted_blocks_data):
    """
    Initialise tous les blocs avec un flag à 0 (pair)
    
    encrypted_blocks_data: dictionnaire contenant les blocs chiffrés
    
    Retourne: dictionnaire mis à jour avec blocs pairs
    """
    pub_key = encrypted_blocks_data['encryption_keys']["public"]
    N, _ = pub_key
    
    print("\n=== INITIALISATION DES FLAGS ===")
    
    # Rendre tous les blocs pairs
    for block_info in encrypted_blocks_data['encrypted_blocks']:
        original = block_info['encrypted_block']
        block_info['encrypted_block'] = _make_block_even(original, N)
    
    return encrypted_blocks_data

def _make_block_even(encrypted_block, N):
    """
    Rend un bloc chiffré pair (flag = 0) via self-blinding
    
    encrypted_block: bloc chiffré (peut être un entier ou des bits)
    N: modulo de Paillier
    
    Retourne: bloc chiffré pair (même type que l'entrée)
    """
    N2 = N ** 2
    
    # Gérer le cas où on reçoit un entier directement
    if isinstance(encrypted_block, (int, type(N))):  # int ou mpz
        block_int = encrypted_block
        return_as_int = True
    else:
        # C'est une liste de bits
        block_int = util.bits_to_int(encrypted_block)
        return_as_int = False
    
    # Si déjà pair, retourner tel quel
    if block_int % 2 == 0:
        return encrypted_block
    
    # Sinon, appliquer self-blinding jusqu'à obtenir un bloc pair
    while True:
        P = paillier.generate_r(N)  # Nombre aléatoire copremier avec N
        P_N = powmod(P, N, N2)
        new_block_int = (block_int * P_N) % N2
        
        if new_block_int % 2 == 0:
            if return_as_int:
                return new_block_int
            else:
                # Reconvertir en bits
                return util.int_to_bits(new_block_int, len(encrypted_block))

def _segment_message(message_bits, alpha):
    """
    Divise un message de 256 bits de taille alpha bits maximum
    
    bits: liste des bits du message
    alpha: taille maximale d'un segment en bits
    
    Retourne une liste de segments (chaque segment est une liste de bits)
    """
    segments = []
    
    for i in range(0, len(message_bits), alpha):
        segment = message_bits[i:i + alpha]
        # Padding avec des 0 si le dernier segment est plus petit que alpha
        if len(segment) < alpha and i + alpha >= len(message_bits):
            segment.extend([0] * (alpha - len(segment)))
        segments.append(segment)
    
    print(f"Message divisé en {len(segments)} segments de {alpha} bits")
    return segments

def _embed_single_message(encrypted_blocks_data, message_segments, start_block_idx):
    """
    Tatoue un seul message dans les blocs à partir de start_block_idx
    
    encrypted_blocks_data: dictionnaire contenant les blocs chiffrés
    message_segments: segments du message (déjà divisé en alpha bits)
    start_block_idx: index du premier bloc à utiliser
    
    Retourne: (encrypted_blocks_data mis à jour, prochain index disponible)
    """
    pub_key = encrypted_blocks_data['encryption_keys']["public"]
    N, _ = pub_key
    blocks = encrypted_blocks_data['encrypted_blocks']
    
    # Vérifier qu'on a assez de blocs
    blocks_needed = len(message_segments) + 1  # +1 pour le bloc de séparation
    if start_block_idx + blocks_needed > len(blocks):
        raise ValueError(f"Pas assez de blocs. Besoin de {blocks_needed}, disponible: {len(blocks) - start_block_idx}")
    
    # Chiffrer les segments
    encrypted_segments = _encrypt_all_message_segments(message_segments, pub_key)
    
    # Tatouer chaque segment
    for i, enc_segment in enumerate(encrypted_segments):
        block_idx = start_block_idx + i
        block_info = blocks[block_idx]
        
        # Tatouer le segment dans le bloc
        watermarked = _embed_segment_in_block(
            block_info['encrypted_block'], 
            enc_segment,  # Passer l'entier directement
            N
        )
        
        # Mettre le flag à 1 (impair)
        watermarked_with_flag = _make_block_odd(watermarked, N)
        
        # Mettre à jour le bloc
        block_info['watermarked_flagged_encrypted_block'] = watermarked_with_flag
    
    # Index du prochain bloc disponible (après le bloc de séparation)
    next_idx = start_block_idx + len(message_segments) + 1
    
    return encrypted_blocks_data, next_idx

def _encrypt_all_message_segments(message_segments, pub_key):
    """
    Chiffre tous les segments d'un message
    
    message_segments: liste de segments de bits
    pub_key: clé publique Paillier
    
    Retourne: liste des segments chiffrés
    """
    encrypted_segments = []
    
    for segment in message_segments:
        encrypted = _encrypt_message_segment(segment, pub_key)
        encrypted_segments.append(encrypted)
    
    return encrypted_segments

def _encrypt_message_segment(message_segment, pub_key):
    """
    Chiffre un segment de message
    
    message_segment: liste de bits de taille alpha
    pub_key: clé publique Paillier
    
    Retourne: segment chiffré
    """
    # Convertir en entier
    segment_int = util.bits_to_int(message_segment)
    
    # Chiffrer
    encrypted_segment = paillier.encrypt(segment_int, pub_key)
    
    return encrypted_segment

def _embed_segment_in_block(encrypted_block, encrypted_segment, N):
    """
    Tatoue un segment chiffré dans un bloc chiffré via multiplication homomorphe
    
    encrypted_block: bloc chiffré (entier ou bits)
    encrypted_segment: segment de message chiffré (entier ou bits)
    N: modulo de Paillier
    
    Retourne: bloc tatoué chiffré (entier)
    """
    N2 = N ** 2
    
    # Convertir en entiers si nécessaire
    if isinstance(encrypted_block, (int, type(N))):
        block_int = encrypted_block
    else:
        block_int = util.bits_to_int(encrypted_block)
    
    if isinstance(encrypted_segment, (int, type(N))):
        segment_int = encrypted_segment
    else:
        segment_int = util.bits_to_int(encrypted_segment)
    
    # Multiplication homomorphe (= addition en clair)
    watermarked_int = (block_int * segment_int) % N2
    
    return watermarked_int

def _make_block_odd(encrypted_block, N):
    """
    Rend un bloc chiffré impair (flag = 1) via self-blinding
    
    encrypted_block: bloc chiffré (peut être un entier ou des bits)
    N: modulo de Paillier
    
    Retourne: bloc chiffré impair (même type que l'entrée)
    """
    N2 = N ** 2
    
    # Gérer le cas où on reçoit un entier directement
    if isinstance(encrypted_block, (int, type(N))):  # int ou mpz
        block_int = encrypted_block
        return_as_int = True
    else:
        # C'est une liste de bits
        block_int = util.bits_to_int(encrypted_block)
        return_as_int = False
    
    # Si déjà impair, retourner tel quel
    if block_int % 2 == 1:
        return encrypted_block
    
    # Sinon, appliquer self-blinding jusqu'à obtenir un bloc impair
    while True:
        P = paillier.generate_r(N)
        P_N = powmod(P, N, N2)
        new_block_int = (block_int * P_N) % N2
        
        if new_block_int % 2 == 1:
            if return_as_int:
                return new_block_int
            else:
                # Reconvertir en bits
                return util.int_to_bits(new_block_int, len(encrypted_block))