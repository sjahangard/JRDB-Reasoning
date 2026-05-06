import os
import pickle
import yaml

from global_functions import *
from graph import *


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


def load_config(path="config.yaml"):
    with open(path, "r") as file:
        return yaml.safe_load(file)


def process_sequence(scene_graph, seq, path_label, path_image, path_label_3d, path_pose):
    print(f"Creating scene graph for sequence: {seq}")

    data_h = read_json_file(path_label, path_image, seq)
    data_h_3d = read_json_file(path_label_3d, path_image, seq)
    data_pose = read_json_file(path_pose, path_image, seq)

    for frame_id in data_h["labels"]:

        frame_key = frame_id.replace("jpg", "pcd")

        if frame_key not in data_h_3d["labels"]:
            continue

        # -------------------------
        # Add human nodes
        # -------------------------
        for i in range(len(data_h["labels"][frame_id])):

            gender, age, race, pose_action, social_group_id, group_size, bbox, human_id, occlusion = \
                extract_person_attributes(data_h["labels"][frame_id][i])

            face_visibility = Face_visibility(
                data_h["labels"][frame_id][i],
                data_pose,
                frame_id
            )

            SR_Robot_Ref, distance_to_robot, bbox_3d = Extract_H_robot_G(
                data_h_3d["labels"][frame_key],
                human_id
            )

            if occlusion not in ["Fully_occluded", "Severely_occluded"]:
                scene_graph.add_human(
                    frame_id=frame_id,
                    age=age,
                    race=race,
                    gender=gender,
                    action=pose_action,
                    social_group_id=social_group_id,
                    group_size=group_size,
                    bbox=bbox,
                    bbox_3d=bbox_3d,
                    human_id_in_frame=human_id,
                    occlusion=occlusion,
                    SR_Robot_Ref=SR_Robot_Ref,
                    distance=distance_to_robot,
                    face_visibility=face_visibility
                )

        # -------------------------
        # Add HHI and HOI physical interactions
        # -------------------------
        for i in range(len(data_h["labels"][frame_id])):

            ID_person, HHI_labels, HOI_labels = Extract_HHI_HOI(
                data_h["labels"][frame_id][i]
            )

            # Human-Human Interaction edges
            for hhi in HHI_labels:
                interaction = list(hhi["inter_labels"].keys())[0]
                pair = hhi["pair"]

                scene_graph.add_physical_relationship(
                    frame_id=frame_id,
                    node_key_1=f"h_{ID_person}",
                    node_key_2=f"h_{pair}",
                    relation_type=interaction
                )

            # Human-Object Interaction edges
            for hoi in HOI_labels:
                interaction = list(hoi["inter_labels"].keys())[0]

                if interaction in None_posed_HUMAN_OBJECT_INTERACTION:
                    Id_track = hoi["pair"][0]
                    Id_category = hoi["pair"][1]

                    scene_graph.add_physical_relationship(
                        frame_id=frame_id,
                        node_key_1=f"h_{ID_person}",
                        node_key_2=f"o_{Id_track}_{Id_category}",
                        relation_type=interaction
                    )

        # -------------------------
        # Add human-human geometrical relationships
        # -------------------------
        for i in range(len(data_h_3d["labels"][frame_key])):

            for j in range(len(data_h_3d["labels"][frame_key])):

                ID_person, ID_person_pair, geometry_relation = Extract_HHG(
                    data_h_3d["labels"][frame_key][i],
                    data_h_3d["labels"][frame_key][j]
                )

                if ID_person != ID_person_pair and "close" in geometry_relation:
                    scene_graph.add_geometric_relationship(
                        frame_id=frame_id,
                        node_key_1=f"h_{ID_person}",
                        node_key_2=f"h_{ID_person_pair}",
                        relation_type=geometry_relation
                    )

    print(f"✅ Scene graph created for sequence: {seq}")
    return scene_graph


def main():
    config = load_config()
    print(config)

    set_ = config["set"]

    path_config = config["paths"][set_]

    path_label = path_config["path_label"]
    path_image = path_config["path_image"]
    path_label_3d = path_config["path_label_3d"]
    path_pose = path_config["path_pose"]
    sequences = path_config.get("sequences", [])

    output_folder = "seq_graphs_test"

    print(f"Running on {set_} set with {len(sequences)} sequences loaded.")

    for seq in sequences:
        scene_graph = SpatialTemporalSceneGraph()

        scene_graph = process_sequence(
            scene_graph=scene_graph,
            seq=seq,
            path_label=path_label,
            path_image=path_image,
            path_label_3d=path_label_3d,
            path_pose=path_pose
        )

        save_scene_graph(scene_graph, seq, output_folder)


if __name__ == "__main__":
    main()