import numpy as np


def normalize_vertices(vertices):
    v_min = np.min(vertices, axis=0)
    v_max = np.max(vertices, axis=0)
    
    v_range = v_max - v_min
    vertices_normalized = (vertices - v_min) / v_range
    
    normalization_params = {
        'v_min': v_min,
        'v_max': v_max,
        'v_range': v_range
    }
    
    return vertices_normalized, normalization_params


def denormalize_vertices(vertices_normalized, normalization_params):
    v_min = normalization_params['v_min']
    v_range = normalization_params['v_range']
    
    vertices = vertices_normalized * v_range + v_min
    
    return vertices


def quantifization(vertices, k=4):
    vertices_work = vertices.copy()
    
    vertices_int = np.round(vertices_work * (10**k)).astype(int)
    vertices_quantified = vertices_int + 10**k

    return vertices_quantified

def dequantization(vertices_quantified, k=4):
    vertices_int = vertices_quantified - 10**k

    vertices = vertices_int.astype(float) / (10**k)

    return vertices

def preprocess_vertices(vertices, k=4):
    preprocessing_info = {'k': k, 'normalize': False}
    
    vertices_work, norm_params = normalize_vertices(vertices)
    preprocessing_info['normalization_params'] = norm_params
    preprocessing_info['normalize'] = True
    vertices_positive = quantifization(vertices_work, k)

    return vertices_positive, preprocessing_info


def inverse_preprocess_vertices(vertices_positive, preprocessing_info):
    k = preprocessing_info['k']
    
    vertices_normalized = dequantization(vertices_positive, k)
    
    if preprocessing_info['normalize']:
        vertices = denormalize_vertices(vertices_normalized, 
                                       preprocessing_info['normalization_params'])
    else:
        vertices = vertices_normalized
    
    return vertices