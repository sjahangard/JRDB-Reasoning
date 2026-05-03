import os
import json
import pickle
from datetime import date

import yaml
from jrdb_reasoning.graph import *
from jrdb_reasoning.graph import SpatialTemporalSceneGraph
from jrdb_reasoning.VG_task import VG_task
from jrdb_reasoning.VQA_task import VQA_task
from jrdb_reasoning.global_functions import (
    create_data_dict,
    choose_question_type,
    update_data_structure_all_slots,
    ensure_folder_exists,
)


def load_config(path="./config.yaml"):
    with open(path, "r") as file:
        return yaml.safe_load(file)


def load_graph(pkl_path):
    with open(pkl_path, "rb") as file:
        import sys
        import jrdb_reasoning.graph as graph

        sys.modules['graph'] = graph
        return pickle.load(file)


def save_json(data, output_folder, file_name):
    ensure_folder_exists(output_folder)

    file_path = os.path.join(output_folder, f"{file_name}.json")
    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=2)

    print(f"✅ Output saved: {file_path}")


def process_vg(scene_graph, config, data_dict, data_points, n_samples):
    all_samples = []

    for _ in range(n_samples):
        data_points = VG_task(
            scene_graph,
            config["S"],
            config["T"],
            config["target_Q"],
            config["modality_type"],
            data_dict,
            data_points,
            config["Incremental"],
        )

        if data_points:
            all_samples.extend(data_points)

    return all_samples


def process_vqa(scene_graph, config, data_dict, data_points):
    question_type = choose_question_type()
    all_samples = []

    if question_type == "Count":
        data_points = VG_task(
            scene_graph,
            config["S"],
            config["T"],
            config["target_Q"],
            config["modality_type"],
            data_dict,
            data_points,
            config["Incremental"],
        )

        updated_samples = update_data_structure_all_slots(data_points, "Count")

        if updated_samples:
            all_samples.extend(updated_samples)

    elif question_type == "Wh":
        print("You selected a WH-type question.")

        data_points = VQA_task(
            scene_graph,
            config["S"],
            config["T"],
            config["target_Q"],
            config["modality_type"],
            data_dict,
            data_points,
            config["Incremental"],
        )

        updated_samples = update_data_structure_all_slots(data_points, "Wh")

        if updated_samples:
            all_samples.extend(updated_samples)

    else:
        print(f"⚠️ Unknown question type: {question_type}")

    return all_samples


def main():
    config = load_config()
    print(config)

    set_name = config["set"]
    task_type = config["task_type"]
    n_samples = config["N"]
    output_folder = config["folder_name"]

    if config["S"] == "1" and config["target_Q"] == "human&object":
        print("⚠️ Warning: 'S=1' and 'target_Q=human&object' cannot be used together.")
        return

    path_config = config["paths"][set_name]
    sequences = path_config.get("sequences", [])

    print(f"Running on {set_name} set with {len(sequences)} sequences loaded.")

    final_data_points = []

    for seq in sequences:
        data_points = []

        data_dict = create_data_dict(
            Set=set_name,
            task_type=task_type,
            S=config["S"],
            T=config["T"],
            target_Q=config["target_Q"],
            modality_type=config["modality_type"],
            seq_name=seq,
        )

        graph_path = f"./seq_graphs/{seq}.pkl"
        scene_graph = load_graph(graph_path)

        if task_type == "VG":
            samples = process_vg(
                scene_graph,
                config,
                data_dict,
                data_points,
                n_samples,
            )

        elif task_type == "VQA":
            samples = process_vqa(
                scene_graph,
                config,
                data_dict,
                data_points,
            )

        else:
            print(f"⚠️ Unknown task type: {task_type}")
            continue

        final_data_points.extend(samples)

        today = date.today()
        file_name = (
            f"{task_type}_T={config['T']}_S={config['S']}_"
            f"{config['target_Q']}_{config['modality_type']}_"
            f"{set_name}_{n_samples}samples_{seq}_{today}"
        )

        save_json(final_data_points, output_folder, file_name)


if __name__ == "__main__":
    main()