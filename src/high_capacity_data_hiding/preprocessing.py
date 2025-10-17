import numpy as np

from src.utils import paillier, util

def preprocess(model: dict[np.array, np.array], config: dict) -> dict:
    k, alpha, encryption_keys = _choose_block_size_and_parameters(config["block_size"])
    pub_key = encryption_keys["public"]
    N, g = pub_key
    vertices = model["vertices"]
    faces = model["faces"]
    b = config["block_size"]

    vertex_blocks = _divide_vertices_into_blocks(vertices, b)

    blocks_data = []
    
    for i, block_vertices in enumerate(vertex_blocks):
        block_bits, metadata = _construct_block(block_vertices.tolist())
        
        watermarkable_block = _create_watermarkable_block(block_bits, k, alpha)
        
        blocks_data.append({
            'block_id': i,
            'original_block': block_bits,
            'watermarkable_block': watermarkable_block,
            'metadata': metadata,
            'vertex_indices': list(range(i*b, (i+1)*b))
        })

    return {
        'blocks_data': blocks_data,
        'n_vertices_original': len(vertices),
        'parameters': {
            'b': b,
            'k': k,
            'alpha': alpha,
            'key_size': pub_key[0].bit_length()
        },
        'encryption_keys': encryption_keys,
        'original_vertices': vertices
    }

def _choose_block_size_and_parameters(b) -> tuple:
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
    N_len = 0
    N_2_len = 0
    while (N_len!=k+1) and N_2_len!=2*k+1:
        encryption_keys = paillier.generate_keys(key_size)
        public_key = encryption_keys["public"]
        N_len = public_key[0].bit_length()
        N_2_len = ((public_key[0])**2).bit_length()
    
    print(f"N_len= {N_len}") 
    print(f"N_2_len= {N_2_len}")    
    return k, alpha, encryption_keys

def _divide_vertices_into_blocks(vertices, b):
    """
    Divise les vertices en blocs de taille b
    
    vertices: tableau numpy de tous les vertices
    b: taille d'un bloc
    
    Retourne: liste de blocs avec padding 
    """
    n_vertices = len(vertices)
    n_blocks = n_vertices // b
    
    blocks = []
    for i in range(n_blocks):
        start_idx = i * b
        end_idx = start_idx + b
        block_vertices = vertices[start_idx:end_idx]
        blocks.append(block_vertices)
    
    # Si il reste des vertices, créer un dernier bloc avec padding
    if n_vertices % b != 0:
        last_block = vertices[n_blocks * b:]
        # Ajouter des vertices (0,0,0) pour compléter
        padding_needed = b - len(last_block)
        padding_vertices = np.zeros((padding_needed, 3))
        last_block_padded = np.vstack([last_block, padding_vertices])
        blocks.append(last_block_padded)
        
    return np.array(blocks)

def _construct_block(vertices):
    """
    Construit un bloc à partir de b vertices selon la méthode HCDH-ED
    
    vertices: liste de b vertices [(x1,y1,z1), (x2,y2,z2), ..., (xb,yb,zb)]
    
    Retourne: 
        - block: liste de bits de taille 69b (où b est le nombre de vertices)
        - metadata: métadonnées pour la reconstruction
    """
    b = len(vertices)
    
    # Extraire toutes les mantisses et métadonnées
    mantisses, metadata = _process_vertices_to_mantissas(vertices)
    
    # Vérifier qu'on a bien 3b mantisses (3 coordonnées par vertex)
    assert len(mantisses) == 3 * b, f"Expected {3*b} mantisses, got {len(mantisses)}"
    
    # Construire le bloc en regroupant les bits par position
    block = []
    
    # Pour chaque position de bit (MSB-0 à LSB)
    for bit_position in range(23):  # 23 bits dans une mantisse
        # Pour chaque vertex
        for vertex_idx in range(b):
            # Pour chaque coordonnée (x, y, z) dans l'ordre
            for coord_idx in range(3):
                mantisse_idx = vertex_idx * 3 + coord_idx
                bit = mantisses[mantisse_idx][bit_position]
                block.append(bit)
    
    # Vérifier la taille du bloc
    assert len(block) == 69 * b, f"Block size should be {69*b}, got {len(block)}"
    
    return np.array(block), metadata

def _process_vertices_to_mantissas(vertices):
    """
    Traite un ensemble de vertices et extrait leurs mantisses et métadonnées
    
    vertices: liste de vertices [(x1,y1,z1), (x2,y2,z2), ..., (xb,yb,zb)]
    
    Retourne: (mantisses, metadata)
        - mantisses: liste des mantisses pour chaque coordonnée
        - metadata: liste des (signe, exposant) pour chaque coordonnée
    """
    mantisses = []
    metadata = []
    
    for vertex in vertices:
        for coord in vertex:  # Pour x, y, z
            mantisse_bits, signe, exposant = _extract_mantissa_sign_exp(coord)
            mantisses.append(mantisse_bits)
            metadata.append((signe, exposant))
    
    mantisses = np.array(mantisses) 
    return mantisses, metadata

def _extract_mantissa_sign_exp(coord):
    """
    Extrait la mantisse, le signe et l'exposant d'une coordonnée flottante
    
    coord: coordonnée flottante
    
    Retourne: (mantisse_bits, signe, exposant)
    """
    # Convertir en float32 pour avoir le format IEEE 754
    float_val = np.float32(coord)
    
    # Obtenir la représentation binaire
    int_repr = float_val.view(np.uint32)
    
    # Extraire le signe (bit 31)
    signe = (int_repr >> 31) & 1
    
    # Extraire l'exposant (bits 23-30)
    exposant = (int_repr >> 23) & 0xFF
    
    # Extraire les 23 bits de mantisse (bits 0-22)
    mantissa = int_repr & 0x7FFFFF
    
    # Convertir la mantisse en liste de bits
    mantisse_bits = util.int_to_bits(mantissa, 23)
    
    return np.array(mantisse_bits), signe, exposant

def _create_watermarkable_block(block_bits, k, alpha):
    """
    Crée un bloc tatouable en mettant les alpha derniers bits des k premiers à zéro
    
    block_bits: bloc de 69b bits
    k: nombre de bits à chiffrer
    alpha: nombre de bits pour le tatouage
    
    Retourne: bloc tatouable
    """
    # Copier le bloc
    watermarkable_block = block_bits.copy()
    
    # Mettre à zéro les alpha derniers bits des k premiers bits
    watermarkable_block = block_bits.copy()
    watermarkable_block[k - alpha:k] = 0  # Plus efficace avec numpy
    
    return np.array(watermarkable_block)