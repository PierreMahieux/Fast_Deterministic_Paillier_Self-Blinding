import random
import os
import numpy as np
import gmpy2
from glob import glob

from src.fast_deterministic_self_blinding import fdsb, embedding, extracting, preprocessing
from src.utils import mesh_utils, util, paillier


QUANTISATION_FACTOR = 4
QIM_STEP = 4
KEY_SIZE = 512
LENGTH_SIGNATURE = 512
LENGTH_MESSAGE = 512

def encrypted_run():
    script_dir = os.path.dirname(__file__)

    dataset_path = "./datasets/meshes/"
    model_name = "casting.obj"
    model_path = os.path.join(script_dir, dataset_path + model_name)

    result_folder = os.path.join(script_dir, f"./results/fdsb/{model_name.split(".")[0]}/")
    result = {}

    model = mesh_utils.load_3d_model(os.path.join(script_dir, model_path))
    watermark_bits = [k%2 for k in range(LENGTH_MESSAGE)]
    random.shuffle(watermark_bits)

    encryption_keys = paillier.generate_keys(KEY_SIZE)
    signing_keys = util.genereate_signing_keys()

    config = {"key_size": KEY_SIZE, "quantisation_factor": QUANTISATION_FACTOR, "qim_step": QIM_STEP, "result_folder": result_folder, "model_path": model_path, "model_name": model_name, "signature_length": LENGTH_SIGNATURE, "message_length": LENGTH_MESSAGE, "save_encrypted_file": True}

    encrypted_vertices, pre_marked_vertices, result_preprocess = fdsb.preprocessing(model["vertices"], encryption_keys, config)
    
    result = result | result_preprocess

    signed_vertices, result_embedding = fdsb.embedding(encrypted_vertices, watermark_bits, {"encryption": encryption_keys, "signing": signing_keys}, config)
    result = result | result_embedding
    signed_vertices = embedding._plain_text_qim_embedding(pre_marked_vertices, watermark_bits, config["qim_step"])

    decrypted_vertices, result_extracting = fdsb.extracting(signed_vertices, {"encryption": encryption_keys, "signing": signing_keys}, config["quantisation_factor"], config["message_length"], config["signature_length"], config["qim_step"])
    result = result | result_extracting

    recovered_mesh = fdsb.recover_mesh(decrypted_vertices, encryption_keys["public"], config["quantisation_factor"])

    if config["save_encrypted_file"]:
        for i in range(len(encrypted_vertices)):
            for j in range(3):
                encrypted_vertices[i][j] = int(gmpy2.digits(encrypted_vertices[i][j])) % (10**config["quantisation_factor"])
        
        mesh_utils.save_3d_model(encrypted_vertices, model["faces"], os.path.join(result_folder, f"encrypted_{model_name}"))
                 

    mesh_utils.save_3d_model(recovered_mesh, model["faces"], os.path.join(result_folder, f"recovered_{model_name}"))

    result["BER_qim"] = util.compare_bits(watermark_bits, result["extracted_watermark"])
    result["config"] = config

    util.write_report(result)

    print("fini")

def plain_text_run(model, qim_delta, model_name):
    
    # model_name = "casting.obj"
    # model_path = os.path.join(script_dir, dataset_path + model_name)

    result_folder = os.path.join(script_dir, f"./results/fdsb/plain_text_qim/delta_{qim_delta}/{model_name.split(".")[0]}/")
    result = {}

    encryption_keys = paillier.generate_keys(KEY_SIZE)
    
    length_message = len(model["vertices"]) * 3
    watermark_bits = [k%2 for k in range(length_message)]
    random.shuffle(watermark_bits)

    config = {"key_size": KEY_SIZE, "quantisation_factor": QUANTISATION_FACTOR, "qim_step": QIM_STEP, "result_folder": result_folder, "signature_length": LENGTH_SIGNATURE, "message_length": length_message, "save_encrypted_file": True, "model_name": model_name}

    quantisation_factor = config["quantisation_factor"]
    qim_step = config["qim_step"]
    vertices = model["vertices"]

    quantized = np.array([[gmpy2.mpz(np.floor(vertices[i_v][i_c] * (10**quantisation_factor))) for i_c in range(3)] for i_v in range(len(vertices))])

    bits = np.array([[gmpy2.powmod(quantized[i][j]//qim_step, 1, 2) for j in range(3)] for i in range(len(quantized))])
    base = (quantized // qim_step) * qim_step
    # pre_marked = quantized.copy()
    pre_marked = np.where(bits == 0, base + qim_step // 2, base - qim_step //2)

    signed_vertices = plain_text_qim_embedding(pre_marked, watermark_bits, config["qim_step"])

    extracted_mark = plain_text_qim_extracting(signed_vertices, len(watermark_bits), config["qim_step"])

    recovered_mesh = recover_mesh(signed_vertices, config["quantisation_factor"])                

    mesh_utils.save_3d_model([[float(quantized[i][j]/(10**config["quantisation_factor"])) for j in range(3)] for i in range(len(quantized))], model["faces"], os.path.join(result_folder, f"quantized_{model_name}"))
    mesh_utils.save_3d_model([[pre_marked[i][j] / (10**config["quantisation_factor"]) for j in range(3)] for i in range(len(pre_marked))], model["faces"], os.path.join(result_folder, f"pre-marked_{model_name}"))
    mesh_utils.save_3d_model(recovered_mesh, model["faces"], os.path.join(result_folder, f"recovered_{model_name}"))

    result["BER_qim"] = util.compare_bits(watermark_bits, extracted_mark)
    result["config"] = config

    util.write_report(result)

def recover_mesh(vertices: np.array, quantisation_factor: int):
    return np.array([[vertices[i][j]/(10**quantisation_factor)for j in range(3)]for i in range(len(vertices))])

def plain_text_qim_extracting(vertices: np.array,length_watermark: int, qim_step: int) -> np.array:
    extracted_W =  np.array([gmpy2.powmod(vertices[i//3][i%3]//qim_step, 1,  2)  for i in range(length_watermark)])
    return extracted_W

def plain_text_qim_embedding(vertices: np.array, watermark: list, qim_step: int) -> np.array:
    bits = np.array([gmpy2.powmod(vertices[i//3][i%3]//qim_step, 1, 2) for i in range(len(watermark))])
    base = (vertices // qim_step) * qim_step
    watermarked = vertices.copy()

    for i in range(len(watermark)):
        if bits[i//3][i%3] == watermark[i]: 
            watermarked[i//3][i%3] = base[i//3][i%3] + qim_step // 2
        else: 
            watermarked[i//3][i%3] = base[i//3][i%3] - qim_step // 2

    return watermarked

if __name__ == "__main__":
    dataset_path = "./datasets/meshes/"
    meshes_list = glob(dataset_path + "*.obj")
    # meshes_list = ["./datasets/meshes/casting.obj"]
    script_dir = os.path.dirname(__file__)
    qim_step_list = [2, 4, 6, 8]
    for model_path in meshes_list:
        model = mesh_utils.load_3d_model(os.path.join(script_dir, model_path))
        for delta in qim_step_list:
            plain_text_run(model, delta, model_path.split('/')[-1])
    print("fini")