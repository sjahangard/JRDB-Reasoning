# pip install transformers==4.52.3

import networkx as nx
import matplotlib.pyplot as plt
from global_functions import *
from graph import *
from VG_task import *
from VQA_task import *
import yaml
from datetime import date
from collections import defaultdict
print(defaultdict)


from Classes import DataDictionary

# Example Usage
scene_graph = SpatialTemporalSceneGraph()

def process_sequence(scene_graph, seq, path_label, path_image, path_pano, path_label_3d, path_pose,pcd_file , bin_file):
    data_h = read_json_file(path_label, path_image, seq)
    # data_o = read_json_file(path_pano, path_image, seq)  # path_pano ---> 2d
    data_h_3d = read_json_file(path_label_3d, path_image, seq)
    # coco = COCO(path_pano + seq)
    data_pose = read_json_file(path_pose, path_image, seq)

    # sampled_numbers = evenly_sample_items(list(data_h['labels'].keys()), N)

    # Processing the data for each frame
    for frame_count, frame_id in enumerate(data_h['labels']):
    # for frame_id in sampled_numbers:
        # Extract the number (assumes it's the numeric part before .jpg)
        number = int(frame_id.split('.')[0])

        ### set human nodes
        for i in range(len(data_h['labels'][frame_id])):
            # Extract 2D person attributes
            gender, age, race, pose_action, social_group_id, group_size, bbox, human_id, occlusion = \
                extract_person_attributes(data_h['labels'][frame_id][i])
            face_visibility = Face_visibility(data_h['labels'][frame_id][i], data_pose, frame_id)
            # print(face_visibility)
            # print(data_h['labels'][frame_id][i]['label_id'])
            # Build key for 3D data and extract robot reference + distance
            frame_key = frame_id.replace('jpg', 'pcd')
            # if frame_key =='000425.pcd':
            #     print('here')
            if frame_key not in data_h_3d['labels']:
                continue
            SR_Robot_Ref, distance_to_robot, bbox_3d = Extract_H_robot_G(data_h_3d['labels'][frame_key],human_id)

            # print(human_id,SR_Robot_Ref, distance_to_robot)

            if occlusion not in ['Fully_occluded', 'Severely_occluded']:
                # Add human node (now including SR_Robot_Ref & distance)
                scene_graph.add_human(
                    frame_id=frame_id,
                    age=age,
                    race=race,
                    gender=gender,
                    action=pose_action,
                    social_group_id=social_group_id,
                    group_size=group_size,
                    bbox=bbox,
                    bbox_3d= bbox_3d,  # added here
                    human_id_in_frame=human_id,
                    occlusion=occlusion,
                    SR_Robot_Ref=SR_Robot_Ref,
                    distance=distance_to_robot,
                    face_visibility= face_visibility
                )

        ### set human-human interaction and human-object interaction edges
        for i in range(len(data_h['labels'][frame_id])):
            ID_person, HHI_labels, HOI_labels = Extract_HHI_HOI(data_h['labels'][frame_id][i])

            # Add HHI edges
            for hhi in HHI_labels:
                interaction = list(hhi['inter_labels'].keys())[0]
                pair = hhi['pair']
                scene_graph.add_physical_relationship(
                    frame_id=frame_id,
                    node_key_1=f'h_{ID_person}',
                    node_key_2=f'h_{pair}',
                    relation_type=interaction
                )

            # Add HOI edges
            for hoi in HOI_labels:
                interaction = list(hoi['inter_labels'].keys())[0]
                if interaction in None_posed_HUMAN_OBJECT_INTERACTION:  ### add Non-pose interactions
                    Id_track = hoi['pair'][0]
                    Id_category = hoi['pair'][1]
                    scene_graph.add_physical_relationship(
                        frame_id=frame_id,
                        node_key_1=f'h_{ID_person}',
                        node_key_2=f'o_{Id_track}_{Id_category}',
                        relation_type=interaction
                    )

        ### add geometrical HHG edges
        pcd = frame_id.replace('jpg', 'pcd')
        # pcd = '000885.pcd'
        if pcd not in data_h_3d['labels']:
                continue
        for i in range(len(data_h_3d['labels'][pcd])):
            ### ADD HHG
            for j in range(len(data_h_3d['labels'][pcd])):
                ID_person, ID_person_pair, geometry_relation = Extract_HHG(data_h_3d['labels'][pcd][i],
                                                                                         data_h_3d['labels'][pcd][j])
                # print('ID_person',ID_person,'ID_person_pair', ID_person_pair,'relation: ', geometry_relation)
                ### add geometrical edges  point : the relation is like {ID_person_pair} is {SR_Person_Ref} of {ID_person]
                if ID_person != ID_person_pair and 'close' in geometry_relation:
                    scene_graph.add_geometric_relationship(frame_id=frame_id, node_key_1=f'h_{ID_person}',
                                                               node_key_2=f'h_{ID_person_pair}', relation_type=geometry_relation)
                    print(f"Frame {frame_id}: {ID_person} to {ID_person_pair} => {geometry_relation}")

    return scene_graph

def load_config(path='config.yaml'):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def main():
    config = load_config()
    print(config)
    set_ = config['set']
    task_type = config['task_type']
    S = config['S']
    T = config['T']
    target_Q = config['target_Q']
    modality_type = config['modality_type']
    N = config['N']
    Incremental = config['Incremental']
    folder_name = config['folder_name']

    # 🚫 Incompatible configuration check
    if S == '1' and target_Q == "human&object":
        print("⚠️ Warning: 'S=1' and 'target_Q=human&object' cannot be used together.")
        return  # Exit the function

    # Load appropriate paths based on train/test
    path_config = config['paths'][set_]
    path_label = path_config['path_label']
    path_image = path_config['path_image']
    path_pano = path_config['path_pano']
    path_label_3d = path_config['path_label_3d']
    path_pose = path_config['path_pose']
    pcd_file = path_config['pcd_file']
    bin_file = path_config['bin_file']
    sequences = path_config.get('sequences', [])

    print(f"Running on {set_} set with {len(sequences)} sequences loaded.")

    data_points = []
    data_points_big = []

    for seq in sequences:
        SpatialTemporalSceneGraph = process_sequence(scene_graph, seq, path_label, path_image, path_pano,
                                                     path_label_3d, path_pose, pcd_file, bin_file)

        data_dict = create_data_dict(Set=set_,
                                     task_type=task_type,
                                     S=S,
                                     T=T,
                                     target_Q=target_Q,
                                     modality_type=modality_type,
                                     seq_name=seq)

        if task_type == 'VG':
            for i in range(N):
                data_points = VG_task(SpatialTemporalSceneGraph, S, T, target_Q, modality_type, data_dict, data_points, Incremental)
                if data_points:
                    data_points_big.extend(data_points)

        elif task_type == 'VQA':

            type_question = choose_question_type()
            type_question = "Wh"

            if type_question == "Count":
                data_points = VG_task(SpatialTemporalSceneGraph, S, T, target_Q, modality_type, data_dict, data_points,
                                      Incremental)
                updated_VQA_samples = update_data_structure_all_slots(data_points,"Count")
                if updated_VQA_samples:
                    data_points_big.extend(updated_VQA_samples)
            elif type_question == "Wh":
                print("You selected a WH-type question.")
                data_points = VQA_task(SpatialTemporalSceneGraph, S, T, target_Q, modality_type, data_dict, data_points, Incremental)
                updated_VQA_samples = update_data_structure_all_slots(data_points, "Wh")
                if data_points:
                    data_points_big.extend(updated_VQA_samples)

    # Save output
    today = date.today()
    special_name = f'{task_type}_T={T}_S={S}_{target_Q}_{modality_type}_{set_}_{N}sample_{today}'
    ensure_folder_exists(folder_name)
    print(special_name)
    file_path = os.path.join(folder_name, f'{special_name}.json')
    with open(file_path, 'w') as json_file:
        json.dump(data_points_big, json_file)

if __name__ == "__main__":
    main()
