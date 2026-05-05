# pip install transformers==4.52.3

import os
import pickle
import yaml
from datetime import date
from collections import defaultdict

import networkx as nx
import matplotlib.pyplot as plt

from global_functions import *
from graph import *
from VG_task import *
from VQA_task import *


# ─── Scene Graph I/O ──────────────────────────────────────────────────────────

def save_scene_graph(scene_graph, seq, folder):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{seq}.pkl")
    with open(path, "wb") as f:
        pickle.dump(scene_graph, f)
    print(f"✅ SceneGraph saved: {path}")


def load_scene_graph(seq, folder):
    path = os.path.join(folder, f"{seq}.pkl")
    with open(path, "rb") as f:
        scene_graph = pickle.load(f)
    print(f"✅ SceneGraph loaded: {path}")
    return scene_graph


def load_graph(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


# ─── Config ───────────────────────────────────────────────────────────────────

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    config = load_config()
    print(config)

    set_          = config["set"]
    task_type     = config["task_type"]
    S             = config["S"]
    T             = config["T"]
    target_Q      = config["target_Q"]
    modality_type = config["modality_type"]
    N             = config["N"]
    Incremental   = config["Incremental"]
    folder_name   = config["folder_name"]

    if S == "1" and target_Q == "human&object":
        print("⚠️  Warning: 'S=1' and 'target_Q=human&object' are incompatible.")
        return

    path_config = config["paths"][set_]
    sequences   = path_config.get("sequences", [])
    print(f"Running on {set_} set with {len(sequences)} sequences.")

    graph_folder   = "seq_graphs"
    data_points_big = []

    for graph_file in os.listdir(graph_folder):
        if not graph_file.endswith(".pkl"):
            continue

        seq_name = graph_file.replace(".pkl", "")
        scene_graph = load_graph(os.path.join(graph_folder, graph_file))

        data_dict = create_data_dict(
            Set=set_, task_type=task_type, S=S, T=T,
            target_Q=target_Q, modality_type=modality_type, seq_name=seq_name,
        )

        data_points = []

        if task_type == "VG":
            for _ in range(N):
                data_points = VG_task(
                    scene_graph, S, T, target_Q, modality_type,
                    data_dict, data_points, Incremental,
                )
            if data_points:
                data_points_big.extend(data_points)

        elif task_type == "VQA":
            type_question = "Wh"  # or choose_question_type()

            if type_question == "Count":
                data_points = VG_task(
                    scene_graph, S, T, target_Q, modality_type,
                    data_dict, data_points, Incremental,
                )
                updated = update_data_structure_all_slots(data_points, "Count")
                if updated:
                    data_points_big.extend(updated)

            elif type_question == "Wh":
                print("Selected WH-type question.")
                data_points = VQA_task(
                    scene_graph, S, T, target_Q, modality_type,
                    data_dict, data_points, Incremental,
                )
                updated = update_data_structure_all_slots(data_points, "Wh")
                if data_points:
                    data_points_big.extend(updated)

        # ── Save output ───────────────────────────────────────────────────────
        today = date.today()
        special_name = (
            f"{task_type}_T={T}_S={S}_{target_Q}_{modality_type}"
            f"_{set_}_{N}samples_{seq_name}_{today}"
        )
        ensure_folder_exists(folder_name)
        file_path = os.path.join(folder_name, f"{special_name}.json")
        with open(file_path, "w") as f:
            json.dump(data_points_big, f)
        print(f"✅ Saved: {file_path}")


if __name__ == "__main__":
    main()