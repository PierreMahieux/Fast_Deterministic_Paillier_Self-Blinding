import os
from gmpy2 import mpz, powmod, invert, gcd, mpz_random, random_state, mpz_urandomb, next_prime
import time
import datetime, hashlib
import numpy as np

from math import ceil

from src.utils import paillier, util, mesh_utils

def run(config: dict, encryption_keys: dict, watermarks: tuple, model):
    print("Run RRDH")
    vertices = model["vertices"]
    faces = model["faces"]
    n_vertices = len(vertices)

    result = {"config": config}

    # Preprocessing
    vertices_prep, prep_info = preprocess_vertices(vertices, config["quantisation_factor"])

    # 2. DIVISION EN PATCHES
    print("\n2. Division en patches...")
    (patches, patch_indices), (isolated_coords, isolated_indices) = divide_into_patches(vertices_prep, faces)
    patch_info = get_patch_info(patches, isolated_coords)

    #génération des clés
    encryption_keys = paillier.generate_keys(config["key_size"])
    pub_key = encryption_keys["public"]
    priv_key = encryption_keys["secret"]
    N, g = pub_key

    # 3. CHIFFREMENT DES PATCHES
    print("\n3. Chiffrement des patches...")
    start_encryption = time.time()
    encrypted_patches, r_values = encrypt_patches(patches, pub_key)
    encrypted_isolated = encrypt_isolated_vertices(isolated_coords, pub_key) if isolated_coords else []
    #recosntruction du modèle chiffré complet
    encrypted_vertices = recover_encrypted_model(
        encrypted_patches, patch_indices, encrypted_isolated, isolated_indices, n_vertices
    )
    result["time_encryption"] = time.time() - start_encryption
    
    # 4. GÉNÉRATION DU WATERMARK
    print("\n4. Génération du watermark...")
    watermark_length = patch_info['n_patches'] * 3
    watermark_original = [np.random.randint(0, 2) for _ in range(watermark_length)]

    
    # 5. TATOUAGE DANS LE DOMAINE CHIFFRÉ
    print("\n5. Tatouage dans le domaine chiffré...")
    start_embedding = time.time()
    watermarked_patches, nb_watermaked_bits = embed_watermark_in_model(
        encrypted_patches, watermark_original, N, config["quantisation_factor"]
    )
    
    # Reconstruction des vertices chiffrés tatoués
    watermarked_encrypted_vertices = recover_encrypted_model(
        watermarked_patches, patch_indices, encrypted_isolated, isolated_indices, n_vertices
    )
    result["time_embedding"] = time.time() - start_embedding
    
    # 6. EXTRACTION DANS LE DOMAINE CHIFFRÉ
    print("\n6. Extraction")
    start_extraction = time.time()
    extracted_watermark = extract_watermark_from_model(
        watermarked_patches, N, 
        expected_length=watermark_length,
        k=config["quantisation_factor"]
    )
    result["time_extraction"] = time.time() - start_extraction
    # Calcul du BER
    ber = util.compare_bits(watermark_original, extracted_watermark)
    result["BER"] = ber
    result["extracted_watermark"] = extracted_watermark

    
    # 7. RESTAURATION DANS LE DOMAINE CHIFFRÉ
    print("\n7. Restauration")
    
    restored_encrypted_patches = restore_encrypted_patches_from_watermarking(watermarked_patches, N, config["quantisation_factor"])
    
    # Reconstruction des vertices chiffrés restaurés
    restored_encrypted_vertices = recover_encrypted_model(
        restored_encrypted_patches, patch_indices, encrypted_isolated, isolated_indices, n_vertices
    )
    # Vertices restaurés déchiffrés
    restored_decrypted_vertices = decrypt_complete_model(restored_encrypted_vertices, priv_key, pub_key)
    # Restauration compléte en appliquant l'inverse du preprocessing
    restored_clear = inverse_preprocess_vertices(restored_decrypted_vertices, prep_info)
    
    # 8. Modèle déchiffré tatoué
    print("\n8. Modèle déchiffré tatoué...")
    
    watermarked_decrypted_vertices = decrypt_complete_model(watermarked_encrypted_vertices, priv_key, pub_key)
    # Inverse preprocessing
    watermarked_clear = inverse_preprocess_vertices(watermarked_decrypted_vertices, prep_info)


    # Sauvegarder les modèles
    # save_3d_model(vertices, faces, os.path.join(result_folder,"original.obj"))
    mesh_utils.save_3d_model(vertices_prep, faces, os.path.join(config["result_folder"],"preprocessed.obj"))
    mesh_utils.save_3d_model(watermarked_clear, faces, os.path.join(config["result_folder"],"watermarked_decrypted.obj"))
    mesh_utils.save_3d_model(restored_clear, faces, os.path.join(config["result_folder"],"restored_decrypted.obj"))
    # mesh_utils.save_3d_model(restored_decrypted_vertices, faces, os.path.join(config["result_folder"],"restored_decrypted.obj"))
    
    # util.write_report(result)
    return result

def preprocess_vertices(vertices, k=4):
    """
    Prétraite des vertices 3D pour le chiffrement Paillier.
    Normalise si nécessaire puis convertit en entiers positifs.
    
    Args:
        vertices: array numpy de shape (N, 3) avec les coordonnées
        k: nombre de chiffres significatifs à conserver (par défaut 4)
        
    Returns:
        vertices_positive: array numpy avec coordonnées entières positives
        preprocessing_info: dict contenant les infos pour l'inversion
    """
    preprocessing_info = {'k': k, 'normalize': False}
    
    # # Normaliser si les valeurs sont hors de [-1, 1]
    # if np.any(vertices < -1) or np.any(vertices > 1):
    #     vertices_positive, norm_params = _normalize_vertices(vertices)
    #     preprocessing_info['normalization_params'] = norm_params
    #     preprocessing_info['normalize'] = True
    # else:
    #     vertices_positive = _quantisation(vertices, k)

    vertices_positive, norm_params = _normalize_vertices(vertices)
    preprocessing_info['normalization_params'] = norm_params
    preprocessing_info['normalize'] = True
    vertices_quantified = _quantisation(vertices_positive, k)
    return vertices_quantified, preprocessing_info

def _normalize_vertices(vertices):
    """
    Normalise les vertices dans l'intervalle [0, 1].
    
    Args:
        vertices: array numpy de shape (N, 3) avec les coordonnées des vertices
        
    Returns:
        vertices_normalized: vertices normalisés dans [0, 1]
        normalization_params: dict contenant les paramètres pour l'inversion
    """
    # Calcul des min et max pour chaque dimension
    v_min = np.min(vertices, axis=0)
    v_max = np.max(vertices, axis=0)
    
    # Éviter la division par zéro
    v_range = v_max - v_min
    #assert v_range == 0, "le modèle a ses valeurs de coordonnées tous égaux "
    
    # Normalisation dans [0, 1]
    vertices_normalized = (vertices - v_min) / v_range
    
    # Sauvegarder les paramètres pour l'inversion
    normalization_params = {
        'v_min': v_min,
        'v_max': v_max,
        'v_range': v_range
    }
    
    return vertices_normalized, normalization_params

def _quantisation(vertices, k=4):
    """
    Quantifie les coordonnées des vertices en conservant k chiffres significatifs.

    Args:
        vertices: array numpy de shape (N, 3) avec les coordonnées
        k: nombre de chiffres significatifs à conserver

    Returns:
        vertices_quantified: array numpy avec coordonnées quantifiées
    """
    vertices_work = vertices.copy()
    
    # 1. Conversion en entiers (multiplication par 10^k)
    vertices_int = np.round(vertices_work * (10**k)).astype(int)
    # 2. Conversion en entiers positifs (ajout de 10^k)
    vertices_quantified = vertices_int + 10**k

    return vertices_quantified

def divide_into_patches(vertices, faces):
    """
    Divise un modèle 3D en patches non-chevauchants.
    
    Args:
        vertices: array numpy (N, 3) des coordonnées
        faces: array numpy (F, 3) des indices des faces
        
    Returns:
        patches: liste d'arrays numpy des patches
        patches_indices: liste des indices pour chaque patch
        isolated_indices: liste des indices des vertices isolés
        isolated_coords: liste des coordonnées des vertices isolés
    """
    n_vertices = len(vertices)
    adjacency = _build_adjacency_graph(faces, n_vertices)
    
    unclassified = set(range(n_vertices))
    classified = set()
    patches_indices = []
    isolated_indices = []
    
    while unclassified:
        # Sélectionner le premier vertex non classé
        seed = min(unclassified)
        
        # Former le patch avec son 2-ring neighborhood complet
        patch_vertices = _get_k_ring_neighbors(seed, adjacency,k=2)
        patch_vertices = patch_vertices - classified
        
        # S'assurer que le patch a au moins 2 vertices
        if len(patch_vertices) >= 2:
            # Créer le patch avec le vertex central en premier
            patch_idx = [seed] + sorted([v for v in patch_vertices if v != seed])
            patches_indices.append(patch_idx)
            
            # Marquer ces vertices comme classés
            classified.update(patch_vertices)
            unclassified -= patch_vertices
        else:
            # Vertex isolé
            isolated_indices.append(seed)
            unclassified.remove(seed)
            classified.add(seed)
    
    # Convertir les indices en arrays de coordonnées
    patches = []
    if patches_indices:
        for indices in patches_indices:
                patch = vertices[indices]
                patches.append(patch)
    
    isolated_coords = []
    if isolated_indices:
        for indices in isolated_indices:
            isolated_coords.append(vertices[indices])
            
    return (patches, patches_indices), (isolated_coords, isolated_indices)

def get_patch_info(patches, isolated_coords=None):
    """
    Obtient des informations sur les patches.
    
    Args:
        patches: liste de patches
        isolated_coords: liste des coordonnées des vertices isolés
        
    Returns:
        dict: statistiques des patches
    """
    sizes = [len(patch) for patch in patches]
    
    info = {
        'n_patches': len(patches),
        'min_size': min(sizes) if sizes else 0,
        'max_size': max(sizes) if sizes else 0,
        'avg_size': np.mean(sizes) if sizes else 0,
        'sizes': sizes,
        'n_isolated': len(isolated_coords) if isolated_coords else 0
    }
    
    return info

def _build_adjacency_graph(faces, n_vertices):
    """
    Construit le graphe d'adjacence à partir des faces.
    
    Args:
        faces: array numpy (F, 3) des indices des faces (indices commençant à 1)
        n_vertices: nombre total de vertices
        
    Returns:
        list: liste d'adjacence pour chaque vertex
    """
    adjacency = [set() for _ in range(n_vertices)]
    
    # Pour chaque face, connecter tous les vertices
    for face in faces:
        # Convertir les indices 1-based en 0-based
        v1, v2, v3 = face[0] - 1, face[1] - 1, face[2] - 1
        adjacency[v1].update([v2, v3])
        adjacency[v2].update([v1, v3])
        adjacency[v3].update([v1, v2])
    
    return adjacency

def _get_k_ring_neighbors(vertex_idx, adjacency, k=2):
    """
    Obtient le voisinage de rang k d'un vertex.
    
    Args:
        vertex_idx: index du vertex central
        adjacency: graphe d'adjacence
        n: rang du voisinage (1-ring, 2-ring, etc.)
        
    Returns:
        set: indices des vertices dans le k-ring neighborhood
    """
    if k == 0:
        return {vertex_idx}
    
    neighbors = {vertex_idx}
    current_ring = {vertex_idx}
    
    for _ in range(k):
        next_ring = set()
        for v in current_ring:
            next_ring.update(adjacency[v])
        current_ring = next_ring - neighbors
        neighbors.update(next_ring)
    
    return neighbors

def encrypt_patches(patches, pub_key):
    """
    Chiffre une liste de patches.
    
    Args:
        patches: liste d'arrays numpy représentant les patches
        pub_key: clé publique (N, g)
        
    Returns:
        encrypted_patches: liste de patches chiffrés
        r_values: liste des valeurs r utilisées pour chaque patch
    """
    N, g = pub_key
    encrypted_patches = []
    r_values = []
    
    for patch in patches:
        # Générer un nouveau r pour chaque patch
        r = paillier.generate_r(N)
        encrypted_patch, _ = encrypt_patch(patch, pub_key, r)
        
        encrypted_patches.append(encrypted_patch)
        r_values.append(r)
    
    return encrypted_patches, r_values

def encrypt_patch(patch, pub_key, r=None):
    """
    Chiffre un patch complet avec le même r.
    
    Args:
        patch: array numpy (Nl, 3) avec coordonnées entières positives
        pub_key: clé publique (N, g)
        r: paramètre aléatoire (si None, en génère un)
        
    Returns:
        encrypted_patch: liste de coordonnées chiffrées
        r: paramètre r utilisé
    """
    N, g = pub_key
    
    # Générer r si non fourni
    if r is None:
        r = paillier.generate_r(N)
    
    # Chiffrer chaque vertex du patch
    encrypted_patch = []
    for vertex in patch:
        encrypted_vertex = _encrypt_vertex(vertex, pub_key, r)
        encrypted_patch.append(encrypted_vertex)
    
    return encrypted_patch, r

def _encrypt_vertex(vertex_coords, pub_key, r):
    """
    Chiffre les coordonnées d'un vertex.
    
    Args:
        vertex_coords: array de 3 coordonnées entières positives [x, y, z]
        pub_key: clé publique (N, g)
        r: paramètre aléatoire pour le chiffrement
        
    Returns:
        list: coordonnées chiffrées [cx, cy, cz]
    """
    encrypted_coords = []
    for coord in vertex_coords:
        c = paillier.encrypt_given_r(int(coord), pub_key, r)
        encrypted_coords.append(c)
    
    return encrypted_coords

def encrypt_isolated_vertices(isolated_coords, pub_key):
    """
    Chiffre les vertices isolés avec des r différents.
    Args:
        isolated_coords: liste d'arrays numpy des coordonnées des vertices isolés
        pub_key: clé publique (N, g)        
    Returns:
        encrypted_isolated_coords: liste des coordonnées chiffrées
    """
    encrypted_isolated_coords = []
    for vertex in isolated_coords:
        r=paillier.generate_r(pub_key[0])
        encrypted_vertex= _encrypt_vertex(vertex, pub_key,r)
        encrypted_isolated_coords.append(encrypted_vertex)
    return encrypted_isolated_coords

def recover_encrypted_model(restored_encrypted_patches, patches_indices, encrypted_isolated_coords, isolated_indices, n_vertices):
    """
    Reconstruit le modèle chiffré complet à partir des patches chiffrés restaurés et vertices chiffrés isolés.
    
    Args:
        restored_encrypted_patches: liste de patches chiffrés restaurés
        patches_indices: indices originaux des vertices dans chaque patch
        encrypted_isolated_coords: coordonnées chiffrées des vertices isolés
        isolated_indices: indices des vertices isolés
        n_vertices: nombre total de vertices dans le modèle
        
    Returns:
        encrypted_vertices: modèle chiffré restauré complet
    """
    # Initialiser le modèle avec des listes vides
    encrypted_vertices = [None] * n_vertices
    
    # Replacer chaque vertex depuis les patches
    for patch, indices in zip(restored_encrypted_patches, patches_indices):
        for vertex_in_patch, original_idx in enumerate(indices):
            encrypted_vertices[original_idx] = patch[vertex_in_patch]
    
    # Replacer les vertices isolés
    if isolated_indices and encrypted_isolated_coords is not None:
        for idx, coords in zip(isolated_indices, encrypted_isolated_coords):
            encrypted_vertices[idx] = coords
    
    return encrypted_vertices

def embed_watermark_in_model(encrypted_patches, watermark_bits, N, k=4):
    """
    Tatoue le watermark dans tous les patches du modèle.
    
    Args:
        encrypted_patches: liste de patches chiffrés
        watermark_bits: bits du watermark
        N: module Paillier
        k: facteur de précision utilisé dans le preprocessing
        
    Returns:
        patches tatoués, nombre de bits tatoués
    """
    watermarked_patches = []
    nb_watermarked_bits = 0
    #valid_patch_indices = []
    
    if len(watermark_bits) < 3 * len(encrypted_patches):        
        print(f"Attention: watermark trop court ({len(watermark_bits)} bits) pour {len(encrypted_patches)} patches")
        watermark_bits = watermark_bits + [0] * (3 * len(encrypted_patches) - len(watermark_bits))
        print(f"  -> complété à {len(watermark_bits)} bits avec des 0")
        
    for i, patch in enumerate(encrypted_patches):
        # Copier le patch
        patch_copy = [vertex[:] for vertex in patch]
        patch_copy = embed_watermark_in_patch(patch_copy, watermark_bits[3*i:3*i+3], N, k)
        watermarked_patches.append(patch_copy)
        nb_watermarked_bits += 3
    return watermarked_patches, nb_watermarked_bits

def embed_watermark_in_patch(encrypted_patch, watermark_bits, N, k=4):
    """
    Tatoue 3 bits dans un patch (1 bit par direction).
    
    Args:
        encrypted_patch: patch chiffré
        watermark_bits: liste de 3 bits [bx, by, bz]
        N: module Paillier
        k: facteur de précision utilisé dans le preprocessing
        
    Returns:
        patch tatoué
    """
    Nl = len(encrypted_patch)
    params = {
        'F_Nl': calculate_F_Nl(Nl, k),
        'T_Nl': calculate_T_Nl(Nl),
        'B_Nl': calculate_B_Nl(Nl, t=50, k=k)
    }
    
    # Calculer les directions originales
    directions_encrypted = compute_all_directions_encrypted(encrypted_patch, N)
    directions = determine_directions_from_encrypted(directions_encrypted, N, Nl, k)
    
    # Tatouer chaque direction
    watermarked_encrypted_patch = encrypted_patch
    for j in range(3):
        if j < len(watermark_bits):
            bit = watermark_bits[j]
            direction = directions[j]
            watermarked_encrypted_patch = embed_bit_in_patch(watermarked_encrypted_patch, bit, j, direction, params, N)
    
    return watermarked_encrypted_patch

def embed_bit_in_patch(encrypted_patch, bit, j, direction, params, N):
    """
    Tatoue un bit dans une direction d'un patch.
    
    Args:
        encrypted_patch: patch chiffré à tatouer
        bit: 0 ou 1 à tatouer
        j: axe (0=x, 1=y, 2=z)
        direction: direction de cet axe
        params: paramètres F(Nl), T(Nl), B(Nl)
        N: module Paillier
        
    Returns:
        patch tatoué (modifié en place)
    """
    if bit == 0:
        # Pas de modification pour le bit 0
        return encrypted_patch
    
    # Déterminer quelles coordonnées modifier
    encrypted_patch = encrypted_patch.copy()  # Pour éviter de modifier l'original
    Nl = len(encrypted_patch)
    
    
    # Bit 1: décaler les coordonnées
    F_Nl = params['F_Nl']
    B_Nl = params['B_Nl']
    N2 = N * N
    
    # Calculer g^B(Nl) mod N² 
    g = N + 1
    g_B = powmod(g, B_Nl, N2)
    
    M =get_M_vector(Nl)
    
    if 0 <= direction <= F_Nl:
        # Direction positive: modifier les vertices avec M(p) = 1
        for i in range(Nl):
            if M[i] == 1:
                old_val = encrypted_patch[i][j]
                encrypted_patch[i][j] = (old_val * g_B) % N2
                
    elif -F_Nl <= direction < 0:
        # Direction négative: modifier le vertex avec M(p) = -1
        for i in range(Nl):
            if M[i] == -1:
                old_val = encrypted_patch[i][j]
                encrypted_patch[i][j] = (old_val * g_B) % N2
                break

    return encrypted_patch

def recover_encrypted_model(restored_encrypted_patches, patches_indices, encrypted_isolated_coords, isolated_indices, n_vertices):
    """
    Reconstruit le modèle chiffré complet à partir des patches chiffrés restaurés et vertices chiffrés isolés.
    
    Args:
        restored_encrypted_patches: liste de patches chiffrés restaurés
        patches_indices: indices originaux des vertices dans chaque patch
        encrypted_isolated_coords: coordonnées chiffrées des vertices isolés
        isolated_indices: indices des vertices isolés
        n_vertices: nombre total de vertices dans le modèle
        
    Returns:
        encrypted_vertices: modèle chiffré restauré complet
    """
    # Initialiser le modèle avec des listes vides
    encrypted_vertices = [None] * n_vertices
    
    # Replacer chaque vertex depuis les patches
    for patch, indices in zip(restored_encrypted_patches, patches_indices):
        for vertex_in_patch, original_idx in enumerate(indices):
            encrypted_vertices[original_idx] = patch[vertex_in_patch]
    
    # Replacer les vertices isolés
    if isolated_indices and encrypted_isolated_coords is not None:
        for idx, coords in zip(isolated_indices, encrypted_isolated_coords):
            encrypted_vertices[idx] = coords
    
    return encrypted_vertices

def extract_bit_from_direction(direction, F_Nl, T_Nl):
    """
    Extrait un bit depuis une direction en claire.
    
    Args:
        direction: valeur de la direction en claire
        F_Nl: F(Nl) du patch
        T_Nl: T(Nl) du patch
        
    Returns:
        0 ou 1
    """
    threshold = F_Nl + T_Nl / 2
    
    if abs(direction) > threshold :#supprimer and abs(direction) < 2*F_Nl + T_Nl
        return 1
    else:
        return 0


def extract_watermark_from_patch(watermarked_encrypted_patch, N, k=4):
    """
    Extrait 3 bits d'un patch chiffré tatoué.
    
    Args:
        encrypted_patch: patch chiffré tatoué
        N: module Paillier
        k: facteur de précision utilisé dans le preprocessing
        
    Returns:
        liste de 3 bits extraits
    """
    #watermarked_encrypted_patch=list((watermarked_encrypted_patch))
    Nl = len(watermarked_encrypted_patch)
   
    F_Nl = calculate_F_Nl(Nl,k)
    T_Nl = calculate_T_Nl(Nl)
    
    # Calculer les directions chiffrées tatouées
    directions_watermarked_encrypted = compute_all_directions_encrypted(watermarked_encrypted_patch, N) 
    
    # Limite pour les directions tatouées
    F_limit = 2 * F_Nl + T_Nl
    
    extracted_bits = []
    for Cdw in directions_watermarked_encrypted:
        direction_w = calculate_direction_from_encrypted(Cdw, N, F_limit)
        bit = extract_bit_from_direction(direction_w, F_Nl, T_Nl)
        extracted_bits.append(bit)
    
    return extracted_bits


def extract_watermark_from_model(watermarked_patches, N, expected_length=None, k=4):
    """
    Extrait le watermark complet du modèle.
    
    Args:
        watermarked_patches: patches tatoués
        N: module Paillier
        expected_length: longueur attendue du watermark
        valid_patch_indices: indices des patches qui ont été tatoués
        k: facteur de précision utilisé dans le preprocessing
        
    Returns:
        bits extraits
    """
    all_bits = []
    
    
    # Extraction depuis tous les patches valides (≥2 vertices)
    for patch in watermarked_patches:
        patch_bits = extract_watermark_from_patch(patch, N, k)
        all_bits.extend(patch_bits)
        
            
    # Tronquer si nécessaire
    if expected_length and len(all_bits) > expected_length:
        all_bits = all_bits[:expected_length]
    
    return all_bits

def restore_encrypted_patch(watermarked_patch, N, k=4):
    """
    Restaure un patch tatoué.
    
    Args:
        watermarked_patch: patch tatoué
        extracted_bits: bits extraits du patch
        N: module Paillier
        k: facteur de précision utilisé dans le preprocessing
        
    Returns:
        patch restauré (patch chiffré)
    """
    Nl = len(watermarked_patch)
    F_Nl = calculate_F_Nl(Nl,k)
    T_Nl = calculate_T_Nl(Nl)
    B_Nl = calculate_B_Nl(Nl, t=50, k=k)
    

    N2 = N * N
    g = N + 1
    
    # Calculer l'inverse de g^B(Nl)
    g_B = powmod(g, B_Nl, N2)
    theta_g_B = invert(g_B, N2)
    
    # Obtenir les directions
    directions_watermarked_encrypted = compute_all_directions_encrypted(watermarked_patch, N)
    F_limit = 2 * F_Nl + T_Nl
    
    M = get_M_vector(Nl)
    threshold = F_Nl + T_Nl / 2
    
    # Copier le patch pour modification
    watermarked_patch = [vertex[:] for vertex in watermarked_patch]
    # Restaurer chaque direction
    for j in range(3):
        Cdw = directions_watermarked_encrypted[j]
        direction = calculate_direction_from_encrypted(Cdw, N, F_limit)
        if abs(direction) > threshold :# bit = 1 and abs(direction) < 2*F_Nl + T_Nl
            
            if direction >= 0:
                # Restaurer les vertices avec M(p) = 1
                for i in range(Nl):
                    if M[i] == 1:
                        old_val = watermarked_patch[i][j]
                        watermarked_patch[i][j] = (old_val * theta_g_B) % N2
            else:
                # Restaurer le vertex avec M(p) = -1
                for i in range(Nl):
                    if M[i] == -1:
                        old_val = watermarked_patch[i][j]
                        watermarked_patch[i][j] = (old_val * theta_g_B) % N2
                        break
    
    return watermarked_patch


def restore_encrypted_patches_from_watermarking(watermarked_patches, N, k=4):
    """
    Restaure les patches chiffrés tatoués.
    
    Args:
        watermarked_patches: patches chiffrés tatoués
        N: module Paillier
        k: facteur de précision utilisé dans le preprocessing
        
    Returns:
        patches chiffrés
    """
    restored_encrypted_patches = []
    
    for patch in watermarked_patches:
        # Copier le patch
        patch_copy = [vertex[:] for vertex in patch]
        
        # Restaurer
        restored_patch = restore_encrypted_patch(patch_copy, N, k)
        restored_encrypted_patches.append(restored_patch)
    
    return restored_encrypted_patches

def recover_encrypted_model(restored_encrypted_patches, patches_indices, encrypted_isolated_coords, isolated_indices, n_vertices):
    """
    Reconstruit le modèle chiffré complet à partir des patches chiffrés restaurés et vertices chiffrés isolés.
    
    Args:
        restored_encrypted_patches: liste de patches chiffrés restaurés
        patches_indices: indices originaux des vertices dans chaque patch
        encrypted_isolated_coords: coordonnées chiffrées des vertices isolés
        isolated_indices: indices des vertices isolés
        n_vertices: nombre total de vertices dans le modèle
        
    Returns:
        encrypted_vertices: modèle chiffré restauré complet
    """
    # Initialiser le modèle avec des listes vides
    encrypted_vertices = [None] * n_vertices
    
    # Replacer chaque vertex depuis les patches
    for patch, indices in zip(restored_encrypted_patches, patches_indices):
        for vertex_in_patch, original_idx in enumerate(indices):
            encrypted_vertices[original_idx] = patch[vertex_in_patch]
    
    # Replacer les vertices isolés
    if isolated_indices and encrypted_isolated_coords is not None:
        for idx, coords in zip(isolated_indices, encrypted_isolated_coords):
            encrypted_vertices[idx] = coords
    
    return encrypted_vertices

def decrypt_complete_model(encrypted_vertices, priv_key, pub_key):
    """
    Déchiffre le modèle complet.
    
    Args:
        encrypted_vertices: liste de vertices chiffrés
        priv_key: clé privée
        pub_key: clé publique
        
    Returns:
        vertices: array numpy du modèle déchiffré
    """    
    decrypted_vertices = []
    
    for encrypted_vertex in encrypted_vertices:
        if encrypted_vertex is not None:
            decrypted_vertex = decrypt_vertex(encrypted_vertex, priv_key, pub_key)
            decrypted_vertices.append(decrypted_vertex)
    
    return np.array(decrypted_vertices)

def decrypt_vertex(encrypted_coords, priv_key, pub_key):
    """
    Déchiffre les coordonnées d'un vertex.
    
    Args:
        encrypted_coords: liste de 3 coordonnées chiffrées
        priv_key: clé privée
        pub_key: clé publique
        
    Returns:
        list: coordonnées déchiffrées [x, y, z]
    """
    decrypted_coords = []
    for c in encrypted_coords:
        m = paillier.decrypt_CRT(c, priv_key, pub_key)
        decrypted_coords.append(m)
    
    return decrypted_coords

def inverse_preprocess_vertices(vertices_positive, preprocessing_info):
    """
    Inverse le preprocessing pour retrouver les coordonnées originales.
    
    Args:
        vertices_positive: vertices avec coordonnées entières positives
        preprocessing_info: dict contenant les infos du preprocessing
        
    Returns:
        vertices: coordonnées originales
    """
    k = preprocessing_info['k']
    
    vertices_normalized = dequantification(vertices_positive, k)
    
    # 3. Dénormaliser si nécessaire
    if preprocessing_info['normalize']:
        vertices = denormalize_vertices(vertices_normalized, 
                                       preprocessing_info['normalization_params'])
    else:
        vertices = vertices_normalized
    
    return vertices

def dequantification(vertices_quantified, k=4):
    """
    Déquantifie les coordonnées des vertices en retrouvant les valeurs d'origine.

    Args:
        vertices_quantified: array numpy avec coordonnées quantifiées
        k: nombre de chiffres significatifs à conserver

    Returns:
        vertices: array numpy avec les coordonnées originales
    """
    # 1. Retirer 10^k
    vertices_int = vertices_quantified - 10**k

    # 2. Diviser par 10^k pour retrouver les décimales
    vertices = vertices_int.astype(float) / (10**k)

    return vertices

def denormalize_vertices(vertices_normalized, normalization_params):
    """
    Inverse la normalisation pour retrouver les coordonnées originales.
    
    Args:
        vertices_normalized: vertices normalisés dans [0, 1]
        normalization_params: paramètres de normalisation
        
    Returns:
        vertices: coordonnées originales
    """
    v_min = normalization_params['v_min']
    v_range = normalization_params['v_range']
    
    vertices = vertices_normalized * v_range + v_min
    
    return vertices

def get_M_vector(Nl):
    """
    Génère le vecteur M pour le calcul des directions.
    
    Args:
        Nl: nombre de vertices dans le patch
        
    Returns:
        array: vecteur M avec M(1)=-1 et M(p>1)=1
    """
    M = np.ones(Nl, dtype=int)
    M[0] = -1
    return M


def calculate_F_Nl(Nl, k=4):
    """
    Calcule F(Nl) qui est égale à d_max (la plus grande direction possible pour un Nl donnée). Le F(Nl) de l'article n'est pas utilisé car il y'a des cas où F(Nl) < d_max. 
    
    Args:
        Nl: nombre de vertices dans le patch
        k: facteur de précision utilisé dans le preprocessing
        
    Returns:
        int: valeur maximale des directions non tatouées
    """
    if Nl <= 1:
        return 0
    
    # Formule originale de l'article
    #F_Nl_original = 1.925 * (Nl-1)**3 - 60.6 * (Nl-1)**2 + 528 * (Nl-1) - 609
    
    # Calculer d_max théorique avec k donné
    # d_max = (Nl-1) * max_coord où max_coord = 2*10^k (après preprocessing)
    d_max = (Nl - 1) * (2 * (10 ** k))
    
    # F(Nl) doit être > d_max avec une marge de sécurité
    F_Nl =  2 * d_max

    return F_Nl


def calculate_T_Nl(Nl, t=50):
    """
    Calcule l'intervalle de robustesse T(Nl).
    
    Args:
        Nl: nombre de vertices dans le patch
        t: facteur de robustesse (par défaut 50)
        
    Returns:
        int: taille de l'intervalle de robustesse
    """
    return t * (Nl - 1)


def calculate_B_Nl(Nl, t=50, k=4):
    """
    Calcule le pas de quantification B(Nl).
    
    Args:
        Nl: nombre de vertices dans le patch
        t: facteur de robustesse
        k: facteur de précision utilisé dans le preprocessing
        
    Returns:
        int: pas de quantification
    """
    if Nl <= 1:
        return 0  # Pas de tatouage possible sur un patch d'un seul vertex
    
    F_Nl = calculate_F_Nl(Nl, k)
    T_Nl = calculate_T_Nl(Nl, t)
    B_Nl = ceil((F_Nl + T_Nl) / (Nl - 1))
    return B_Nl





def compute_encrypted_direction(encrypted_patch, j, N):
    """
    Calcule la direction chiffrée Cd pour l'axe j d'un patch.
    
    Args:
        encrypted_patch: patch chiffré [[c_x, c_y, c_z], ...]
        j: axe (0=x, 1=y, 2=z)
        N: module du système Paillier
        
    Returns:
        mpz: direction chiffrée Cd
    """
    N2 = N * N
    Nl = len(encrypted_patch)
    M = get_M_vector(Nl)
    
    # Initialiser le produit
    Cd = mpz(1)
    
    # Calculer le produit selon M
    for i in range(Nl):
        coord_encrypted = encrypted_patch[i][j]
        
        if M[i] == 1:
            # Multiplier par C[i,j]
            Cd = (Cd * coord_encrypted) % N2
        else:  # M[i] == -1
            # Multiplier par C[i,j]^(-(Nl-1))
            C_inv = invert(coord_encrypted, N2)
            C_inv_power = powmod(C_inv, Nl-1, N2)
            Cd = (Cd * C_inv_power) % N2
    
    return Cd

def calculate_direction_from_encrypted(Cd, N, F_limit):
    """
    Calcule la direction d à partir de son chiffré Cd.
    Utilise le fait que g = N+1.
    
    Args:
        Cd: direction chiffrée
        N: module du système Paillier
        F_limit: limite pour déterminer le signe (F(Nl) ou 2F(Nl)+T(Nl))
        
    Returns:
        int: direction en clair
    """
    # Calculer d mod N
    d_mod_N = ((Cd - 1) // N) % N
    
    # Déterminer le signe
    if d_mod_N <= F_limit and d_mod_N >= 0:
        # Direction positive
        d = int(d_mod_N)
    elif d_mod_N >= N - F_limit and d_mod_N <= N-1:
        # Direction négative
        d = int(d_mod_N) - N
    else:
        # Cas imprévu
        d = 0
        print(f"Alerte: direction mod [N] calculée est hors intervalle. Voir dans le fichier directions.py à la fonction calculate_direction_from_encrypted.")
    return d


def compute_all_directions_encrypted(encrypted_patch, N):
    """
    Calcule toutes les 3 directions chiffrées d'un patch chiffré.
    
    Args:
        encrypted_patch: patch chiffré
        N: module du système Paillier
        
    Returns:
        list: [Cd_x, Cd_y, Cd_z]
    """
    directions_encrypted = []
    
    for j in range(3):  # x, y, z
        Cd = compute_encrypted_direction(encrypted_patch, j, N)
        directions_encrypted.append(Cd)
    
    return directions_encrypted


def determine_directions_from_encrypted(directions_encrypted, N, Nl, k):
    """
    Détermine les directions en clair à partir des directions chiffrées.
    
    Args:
        directions_encrypted: [Cd_x, Cd_y, Cd_z]
        N: module du système Paillier
        Nl: nombre de vertices dans le patch
        
    Returns:
        list: [d_x, d_y, d_z] directions en clair
    """
    F_Nl = calculate_F_Nl(Nl, k)
    directions = []
    
    for Cd in directions_encrypted:
        d = calculate_direction_from_encrypted(Cd, N, F_Nl)
        directions.append(d)
    
    return directions


def compute_directions_cleartext(patch, j):
    """
    Calcule la direction en clair pour vérification.
    
    Args:
        patch: patch en clair (Nl, 3)
        j: axe (0=x, 1=y, 2=z)
        
    Returns:
        int: direction en clair
    """
    Nl = len(patch)
    M = get_M_vector(Nl)
    
    d = 0
    for i in range(Nl):
        if M[i] == 1:
            d = d + patch[i, j] * M[i]
        else:  # M[i] == -1
            d = d - (Nl-1) * patch[i, j]
            
    return int(d)


def get_watermarking_params(Nl, t=50, k=4):
    """
    Obtient tous les paramètres de tatouage pour un patch.
    
    Args:
        Nl: nombre de vertices dans le patch
        t: facteur de robustesse
        k: facteur de précision utilisé dans le preprocessing
        
    Returns:
        dict: paramètres F(Nl), T(Nl), B(Nl)
    """
    params = {
        'F_Nl': calculate_F_Nl(Nl, k),
        'T_Nl': calculate_T_Nl(Nl, t),
        'B_Nl': calculate_B_Nl(Nl, t, k),
        'Nl': Nl,
        't': t,
        'k': k
    }
    
    return params