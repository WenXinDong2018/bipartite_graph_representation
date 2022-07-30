import subprocess
import os
from scipy.sparse import load_npz
from src.learning.generate.graph_utils import convert_adj_to_interaction

def box(data_path, dim = 4, box_type = "tbox", config={}):

    folder_name = f"dim={dim}"
    for key, value in config.items():
        folder_name = f"{folder_name}_{key}={value}"

    output_dir = os.path.join(data_path, "results", box_type, folder_name)

    command = f"graph_modeling train --data_path={data_path} --output_dir={output_dir} --model_type={box_type} --save_prediction --dim={dim}"

    for key, value in config.items():
        command += f" --{key}={value}"

    subprocess.call(command, shell=True)

    prediction_file = ""
    for folder in os.listdir(output_dir):
        for file in os.listdir(os.path.join(output_dir, folder)):
            if file.endswith(".npz"):
                prediction_file = os.path.join(output_dir, folder, file)
                break

    predicted_adj = load_npz(prediction_file)

    #convert adj to interaction matrix for fair comparison
    predicted_interaction_matrix = convert_adj_to_interaction(predicted_adj, data_path)

    return predicted_interaction_matrix
