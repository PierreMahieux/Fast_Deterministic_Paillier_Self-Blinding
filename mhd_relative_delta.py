import os
import random
import numpy as np
from glob import glob
import json
import pymeshlab
import matplotlib.pyplot as plt

from src.utils import mesh_utils

if __name__ == "__main__":

    dataset_path = "./datasets/meshes/"
    meshes_list = glob(dataset_path + "*.obj")
    # meshes_list = [os.path.join(dataset_path, "casting.obj")]

    qim_step_list = [2, 4, 6, 8]
    hausdorff = {}
    for mesh in meshes_list:
        hausdorff[mesh.split('/')[-1].split('.')[0]] = []
        print(f"file {mesh.split("/")[-1]}")
        
        for qim_step in qim_step_list:
            result_folder_path = f"/home/pierremahieux/Documents/Projets/Fast_Paillier_Deterministic_Self_Blinding/results/fdsb/plain_text_qim/delta_{qim_step}/{mesh.split("/")[-1].split(".")[0]}"

            result_file = open(os.path.join(result_folder_path, "report.txt"), 'r')
            result = json.loads(result_file.read())
            result_file.close()    

            meshset = pymeshlab.MeshSet()

            meshset.load_new_mesh(os.path.join(result["config"]["result_folder"], f"recovered_{result["config"]["model_name"]}"))
            model_1_mesh = meshset.mesh(0)

            meshset.load_new_mesh(os.path.join(result["config"]["result_folder"], f"quantized_{result["config"]["model_name"]}"))
            model_2_mesh = meshset.mesh(1)

            hausdorff[mesh.split('/')[-1].split('.')[0]].append(np.max([meshset.apply_filter('get_hausdorff_distance', targetmesh=1, sampledmesh=0)["max"],meshset.apply_filter('get_hausdorff_distance', targetmesh=0, sampledmesh=1)["max"]]))


    markers = ['1', '2', '3', '4', '+']
    i_m=0
    plt.figure(figsize=(10, 5), layout='constrained')
    plt.xlabel(r"$\Delta$", fontsize=17, usetex=True)
    plt.ylabel("Hausdorff distance", fontsize=17)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
    plt.ticklabel_format(axis='x', useMathText=True)
    plt.xticks(qim_step_list, fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(True)
    for key, value in hausdorff.items():
        plt.plot(qim_step_list, value, label=key, marker=markers[i_m], markersize=10)
        i_m += 1
    plt.legend(prop={'size': 15})
    # plt.show()

    plt.savefig(f"/home/pierremahieux/Documents/Communications/Diagrammes/FDSB/mhd_delta-qim_preprocessed_models.pdf")
    print("fini")