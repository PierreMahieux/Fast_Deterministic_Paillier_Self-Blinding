# Fast Deterministic Self-Blinding for Metadata-Free Integrity Verification of Paillier Encrypted Data: Application to 3D Models

Ensuring the integrity and authenticity of encrypted data without relying on auxiliary metadata remains a critical challenge in secure cloud-based processing. Existing schemes based on the probabilistic self-blinding property of the Paillier cryptosystem have been used to embed fragile watermarks in the encrypted domain for tamper detection. However, such approaches are not fully secure, as their probabilistic nature requires multiple random trials for successful embedding and cannot achieve complete integrity verification without storing digital signatures or hashes in separate files. In this paper, we introduce a fast deterministic self-blinding scheme that overcomes these limitations by enabling direct, metadata-free storage of integrity information within the ciphertext itself. Our method deterministically modulates encrypted values in constant time while preserving the semantic security of the Paillier cryptosystem. This property allows embedding a cryptographic signature of the encrypted data directly into the ciphertext, ensuring integrity and authenticity verification without decryption or auxiliary storage. Furthermore, by integrating Quantization Index Modulation (QIM) watermarking, our framework supports robust traceability of the decrypted data. Experiments on 3D models demonstrate that the proposed scheme maintains high visual quality, achieves real-time performance, and provides complete integrity/authenticity control in the encrypted domain.

## Overview 

This project proposes a scheme to watermark and control the integrity of 3D models. We embed a watermark using a QIM embedding scheme with the homomorphic addition and then embed a signature in the model using a deterministic self-blinding method. This allow for a full integrity control and traceability of the model. The self-blinding signature embedding is made possible by our deterministic approach as well as being faster than probabilistic approach from the litterature to embed a given message.

## Requirements

This project requires :
- gmpy2
- numpy
- matplotlib
- scipy
- ecdsa
- pymeshlab

## Setup

1. Clone the repository
2. Create a virtual environment
```
python -m venv venv
```
3. Activate the environment
```
source venv/bin/activate
```
4. Install the required packages
```
pip install -r requirements.txt
```


## Running the code

To run our methods on every models from the dataset, files are available in ./dataset/meshes/, use the provided *configs/full_evaluation.json* configuration file :
```
python run_evaluation.py --config_path configs/full_evaluation.json
```
This configuration will run with different QIM parameters as well as 
We provide 1 other configuration file to quickly test our method on the *casting* object.

### Configuration files

The configuration file should be a json file with these variables :
- key_size : controls the bit size of the Paillier key, generally 1024 or 2048 bits;
- quantisation_factor: the quantisation factor to use during the preprocessing, 4 is considered enough;
- message_length: the length in bits of the message to be hidden, "max" sets the length of the message to the maximum capacity of the models;
- list_qim_step: the list of different QIM quantization step to run;
- qim_step: the default QIM quantization step value if no list is provided;
- save_encrypted_file: wheter you want to save the 3D model after encryption;
- models: list of the models to watermark (*e.g.* ["casting.obj"] to watermark the *casting* object), or "all" to run the code on every object located in the *dataset/meshes* folder;
- methods: list of the methods to run (*e.g.* ["rrdh"] to run the base method), "all" to run both methods.

## License

This project is licensed under the terms of the MIT license.
