import os
import random
import numpy as np
from glob import glob
import json
import matplotlib.pyplot as plt

from src.utils import mesh_utils

if __name__ == "__main__":
    dataset_path = "./datasets/meshes/"
    meshes_list = glob(dataset_path + "*.obj")

    noise_level = [0, 2e-5, 5e-5, 1e-4, 2e-4, 3e-4, 4e-4]
    delta_qim_list = [2, 4, 6, 8]
    delta_qim_list = [4]
    
    for delta_qim in  delta_qim_list:

        ber = {}
        for mesh in meshes_list:
            ber[mesh.split('/')[-1].split('.')[0]] = []

            result_folder_path = f"/home/pierremahieux/Documents/Projets/Fast_Paillier_Deterministic_Self_Blinding/results/fdsb/{mesh.split("/")[-1].split(".")[0]}/delta_{delta_qim}"

            result_file = open(os.path.join(result_folder_path, "report.txt"), 'r')
            result = json.loads(result_file.read())
            result_file.close()

            model = mesh_utils.load_3d_model(os.path.join(result["config"]["result_folder"], f"recovered_{result["config"]["model_name"]}"))
            for noise_step in noise_level:

                noise = np.random.normal(0, noise_step, model["vertices"].shape)

                noisy_vertices = model["vertices"] + noise
                mesh_utils.save_3d_model(noisy_vertices, model["faces"], os.path.join(result["config"]["result_folder"], f"noisy_sigma_{noise_step}_{result["config"]["model_name"]}"))
                
                noisy_watermark = []
                for i in range(result["config"]["message_length"]):
                    noisy_watermark.append(int((noisy_vertices[i//3][i%3] * 10**result["config"]["quantisation_factor"]//result["config"]["qim_step"]) % 2))

                ber[mesh.split('/')[-1].split('.')[0]].append(np.mean(np.array(result["watermark_embedded"]) != noisy_watermark))

        markers = ['1', '2', '3', '4', '+']
        i_m=0
        plt.figure(figsize=(10, 5), layout='constrained')
        plt.xlabel(f"Gaussian Noise Level with delta = {delta_qim}", fontsize=17)
        plt.ylabel("Bit Error Rate", fontsize=17)
        plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0), useMathText=True)
        # plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        # plt.xscale('log')
        plt.xticks(noise_level, fontsize=15)
        plt.yticks([0, 0.3, 0.5, 0.7, 1], fontsize=15)
        plt.ylim(0, 1)
        plt.xlim(0, noise_level[-1])
        plt.grid(True)
        for key, value in ber.items():
            plt.plot(noise_level, value, label=key, marker=markers[i_m], markersize=10)
            i_m += 1
        plt.legend(prop={'size': 15})
        # plt.text(0, 0, f"delta = {delta_qim}", fontsize=13)#, transform=fig.transFigure)
        plt.show()
        # plt.savefig(f"/home/pierremahieux/Documents/Communications/Diagrammes/FDSB/ber_gaussian_noise_delta_{delta_qim}.pdf")

    print("fini")