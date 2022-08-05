import subprocess
import os
import json
from scipy.sparse import load_npz, save_npz
from src.learning.generate.graph_utils import convert_adj_to_interaction, NpEncoder
from src.learning.training.evaluate import compare_interaction_matrices

def box(data_path, dim = 4, box_type = "tbox", config={}):

    #directory for saving experiment
    folder_name = f"dim={dim}"
    for key, value in config.items():
        folder_name = f"{folder_name}_{key}={value}"

    output_dir = os.path.join(data_path, "results", f"n_params={dim*2}", box_type, folder_name)
    metrics_path = os.path.join(output_dir, "metrics.json")
    predicted_interaction_matrix_path = os.path.join(output_dir, "interaction_matrix")

    #if already exist simply return existing metrics
    if os.path.exists(output_dir):
        prediction_coo = load_npz(os.path.join(output_dir, "interaction_matrix.npz"))
        metrics = json.load(open(metrics_path, "r"))
        return metrics, prediction_coo

    # train
    command = f"graph_modeling train --data_path={data_path} --output_dir={output_dir} --model_type={box_type} --save_prediction --dim={dim}"
    for key, value in config.items():
        command += f" --{key}={value}"
    subprocess.call(command, shell=True)

    # save
    prediction_file = ""
    for folder in os.listdir(output_dir):
        if os.path.isdir(os.path.join(output_dir, folder)):
            for file in os.listdir(os.path.join(output_dir, folder)):
                if file.endswith(".npz"):
                    prediction_file = os.path.join(output_dir, folder, file)
                    break

    predicted_adj = load_npz(prediction_file)

    #convert adj to interaction matrix
    predicted_interaction_matrix = convert_adj_to_interaction(predicted_adj, data_path)
    metrics = compare_interaction_matrices(data_path, predicted_interaction_matrix)
    os.makedirs(output_dir, exist_ok=True)

    json.dump(metrics, open(metrics_path, "w"), cls = NpEncoder)

    save_npz(predicted_interaction_matrix_path, predicted_interaction_matrix)
    return metrics, predicted_interaction_matrix
