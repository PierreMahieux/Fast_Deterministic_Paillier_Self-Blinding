import numpy as np


def build_adjacency_graph(faces, n_vertices):
    adjacency = [set() for _ in range(n_vertices)]
    
    for face in faces:
        v1, v2, v3 = face[0] - 1, face[1] - 1, face[2] - 1
        adjacency[v1].update([v2, v3])
        adjacency[v2].update([v1, v3])
        adjacency[v3].update([v1, v2])
    
    return adjacency


def get_k_ring_neighbors(vertex_idx, adjacency, k=2):
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


def divide_into_patches(vertices, faces):
    n_vertices = len(vertices)
    adjacency = build_adjacency_graph(faces, n_vertices)
    
    unclassified = set(range(n_vertices))
    classified = set()
    patches_indices = []
    isolated_indices = []
    
    while unclassified:
        seed = min(unclassified)
        
        patch_vertices = get_k_ring_neighbors(seed, adjacency,k=2)
        patch_vertices = patch_vertices - classified
        
        if len(patch_vertices) >= 2:
            patch_idx = [seed] + sorted([v for v in patch_vertices if v != seed])
            patches_indices.append(patch_idx)
            
            classified.update(patch_vertices)
            unclassified -= patch_vertices
        else:
            # Vertex isolé
            isolated_indices.append(seed)
            unclassified.remove(seed)
            classified.add(seed)
    
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

