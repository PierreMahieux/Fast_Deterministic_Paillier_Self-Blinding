import os
import numpy as np
import matplotlib.pyplot as plt
import trimesh
from mpl_toolkits.mplot3d import Axes3D
import time,random
from datetime import datetime
import plotly.graph_objects as go
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def load_3d_model(filename=None):
    """Charge un modèle 3D depuis un fichier .obj ou génère un cube simple
       filename: nom du fichier .obj à charger
       
       Retourne les noeuds et les faces en tableau numpy"""
    
    # Charger depuis un fichier .obj
    vertices = []
    faces = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('v '):  # Vertex
                parts = line.split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                vertices.append([x, y, z])
            
            elif line.startswith('f '):  # Face
                parts = line.split()
                face_indices = []
                for part in parts[1:]:
                    vertex_index = int(part.split('/')[0]) - 1
                    face_indices.append(vertex_index)
                
                if len(face_indices) == 3:
                    faces.append(face_indices)
                elif len(face_indices) == 4:
                    faces.append([face_indices[0], face_indices[1], face_indices[2]])
                    faces.append([face_indices[0], face_indices[2], face_indices[3]])
        
        vertices = np.array(vertices, dtype=float)
        faces = np.array(faces) if faces else None
        
        print(f"Fichier {filename.split("/")[-1]} chargé avec succès")
        print(f"  Nombre de vertices: {len(vertices)}")
        print(f"  Nombre de faces: {len(faces) if faces is not None else 0}")
        
        return {"vertices":vertices, "faces":faces}

def save_3d_model(vertices, faces, filename):
    """Sauvegarde un modèle 3D dans un fichier .obj"""
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as file:
            for vertex in vertices:
                file.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
            file.write("\n")
            
            if faces is not None:
                for face in faces:
                    file.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        
        print(f"Modèle sauvegardé dans {filename}")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde: {e}")  
        
def visualize_3d_model_trimesh(vertices, faces, title="Modèle 3D", save_path=None):
    """Visualise un modèle 3D avec trimesh en png, .jpeg"""
    
    # S'assurer que les types sont corrects
    vertices = np.array(vertices, dtype=np.float64)
    faces = np.array(faces, dtype=np.int32)
    
    # CORRECTION : Retirer les dimensions supplémentaires
    if vertices.ndim > 2:
        print(f"⚠️ Correction de la forme des vertices: {vertices.shape} → ", end="")
        vertices = vertices.squeeze()  # Retire les dimensions de taille 1
        print(f"{vertices.shape}")
    
    # Vérifier que c'est bien (N, 3)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        print(f"❌ Forme incorrecte des vertices: {vertices.shape}")
        return
    
    # Vérifier les dimensions
    print(f"Debug - Vertices shape: {vertices.shape}, Faces shape: {faces.shape}")
    print(f"Debug - Faces min: {faces.min()}, max: {faces.max()}")
    
    # Créer le mesh SANS traitement automatique
    mesh = trimesh.Trimesh(vertices=vertices, 
                          faces=faces, 
                          process=False,
                          validate=False)
    
    # Sauvegarder si demandé
    if save_path:
        try:
            scene = trimesh.Scene(mesh)
            png = scene.save_image(resolution=[1920, 1080])
            with open(save_path, 'wb') as f:
                f.write(png)
            print(f"Modèle sauvegardé: {save_path}")
        except Exception as e:
            print(f"Impossible de sauvegarder: {e}")
    
    # Afficher le mesh
    mesh.show(caption=title)
    
def visualize_3d_model_trimesh(vertices, faces, title="Modèle 3D", save_path=None):
    """Visualise un modèle 3D avec trimesh en png, .jpeg"""
    # Créer le mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Sauvegarder si demandé
    if save_path:
        scene = mesh.scene()
        # Prendre une capture d'écran
        png = scene.save_image(resolution=[1920, 1080])
        with open(save_path, 'wb') as f:
            f.write(png)
    
    print(f"Modèle sauvegardé: {save_path}")

    ## Afficher le mesh
    mesh.show(caption=title)  
                        
def prepare_for_visualization(encrypted_vertices, N):
    """
    Prépare les vertices chiffrés ou ceux qui ont de garnds coordonnées pour la visualisation. 
    La fonction réduit la taille des valeurs chiffrés en gardant les caractéristiques"""
    visual_vertices = []
    for vertex in encrypted_vertices:
        visual_vertex = []
        for coord in vertex:
            ## Réduire en gardant les caractéristiques
            reduced = int(coord // (N // 10000))
            visual_coord = (reduced % 4000) - 2000
            visual_vertex.append(visual_coord)
        visual_vertices.append(visual_vertex)
    
    vertices_array = np.array(visual_vertices, dtype=float)
    # Normaliser pour garder la forme du cube
    max_val = np.max(np.abs(vertices_array))
    if max_val > 0:
        vertices_array = vertices_array / max_val * 2
    
    return vertices_array