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
# class SpatialTemporalSceneGraph:
#     def __init__(self):
#         self.graph = nx.DiGraph()  # Directed graph for spatial-temporal relationships
#         self.frame_data = {}  # Stores data per frame
#
#     def add_human(self, frame_id, age, race, gender, action, social_group_id, group_size, bbox, human_id_in_frame, occlusion):
#         """
#         Add a human node to a specific frame.
#         - frame_id: Frame number.
#         - age, race, gender, action: Human attributes.
#         - social_group_id, group_size: Social group identifier and size.
#         - bbox: Bounding box in the format (x1, y1, x2, y2).
#         - human_id_in_frame: Identifier for the human within the frame.
#         - occlusion: Occlusion status (e.g., True/False or a value between 0 and 1).
#         """
#         if frame_id not in self.frame_data:
#             self.frame_data[frame_id] = {}
#
#         # Use a unique key for human nodes
#         key = f"h_{human_id_in_frame}"
#         self.frame_data[frame_id][key] = {
#             'type': 'human',
#             'age': age,
#             'race': race,
#             'gender': gender,
#             'action': action,
#             'social_group_id': social_group_id,
#             'group_size': group_size,
#             'bbox': bbox,
#             'occlusion': occlusion  # Add occlusion to frame data
#         }
#
#         # Node name in the graph includes the frame and the unique key
#         node_name = f"{frame_id}_{key}"
#         self.graph.add_node(node_name,
#                             node_type='human',
#                             age=age,
#                             race=race,
#                             gender=gender,
#                             action=action,
#                             social_group_id=social_group_id,
#                             group_size=group_size,
#                             bbox=bbox,
#                             ID=human_id_in_frame,
#                             occlusion=occlusion)  # Include occlusion as an attribute
#
#     def add_object(self, frame_id, category, object_id_in_frame, bbox=None):
#         """
#         Add an object node to a specific frame.
#         - frame_id: Frame number.
#         - category: Category of the object (e.g., 'car', 'chair').
#         - object_id_in_frame: Identifier for the object within the frame.
#         - bbox: (Optional) Bounding box (x1, y1, x2, y2) for visualization.
#                 If not provided, a default bbox is used.
#         """
#         if frame_id not in self.frame_data:
#             self.frame_data[frame_id] = {}
#
#         # Use a unique key for object nodes
#         key = f"o_{object_id_in_frame}"
#         if bbox is None:
#             # Set a default bbox for object nodes if not provided.
#             bbox = (50, 50, 70, 70)
#
#         self.frame_data[frame_id][key] = {
#             'type': 'object',
#             'category': category,
#             'bbox': bbox
#         }
#
#         node_name = f"{frame_id}_{key}"
#         self.graph.add_node(node_name,
#                             node_type='object',
#                             category=category,
#                             bbox=bbox,
#                             ID=object_id_in_frame)
#
#     def add_relationship(self, frame_id, node_key_1, node_key_2, relation_type):
#         """
#         Add a spatial relationship between two nodes in the same frame.
#         - frame_id: Frame number.
#         - node_key_1: The unique key for the first node (e.g., 'h_1' or 'o_1').
#         - node_key_2: The unique key for the second node.
#         - relation_type: Type of relation (e.g., 'next_to', 'moving_towards').
#         """
#         self.graph.add_edge(f"{frame_id}_{node_key_1}",
#                             f"{frame_id}_{node_key_2}",
#                             relation_type=relation_type)
#
#     def add_temporal_relationship(self, frame_id_1, node_key_1, frame_id_2, node_key_2, relation_type):
#         """
#         Add a temporal relationship between nodes across frames.
#         - frame_id_1: Earlier frame number.
#         - node_key_1: Unique key for node in frame 1.
#         - frame_id_2: Later frame number.
#         - node_key_2: Unique key for node in frame 2.
#         - relation_type: Type of temporal relation (e.g., 'followed_by').
#         """
#         self.graph.add_edge(f"{frame_id_1}_{node_key_1}",
#                             f"{frame_id_2}_{node_key_2}",
#                             relation_type=relation_type)
#
#     def get_object_at_frame(self, frame_id):
#         """
#         Retrieve all nodes (humans and objects) for a specific frame.
#         """
#         return self.frame_data.get(frame_id, {})
#
#     def get_relationships(self):
#         """
#         Get all relationships in the graph.
#         """
#         return self.graph.edges(data=True)
#
#     def get_temporal_relationships(self):
#         """
#         Get all temporal relationships in the graph.
#         """
#         temporal_edges = [(u, v, data) for u, v, data in self.graph.edges(data=True)
#                           if 'frame' in u and 'frame' in v]
#         return temporal_edges
#
#     def visualize_frame(self, frame_id):
#         """
#         Visualize a specific frame with its nodes (humans and objects) and spatial relationships.
#         Node labels include attributes:
#           - For humans: ID, Age, Race, Gender, Action, Social Group ID, Group Size.
#           - For objects: ID, Category.
#         The node positions are determined by the top-left corner of their bbox.
#         """
#         frame_nodes = self.get_object_at_frame(frame_id)
#         # Build the list of node names from the frame data.
#         node_names = [f"{frame_id}_{key}" for key in frame_nodes.keys()]
#         subgraph = self.graph.subgraph(node_names)
#
#         plt.figure(figsize=(8, 6))
#
#         # Define node positions: use the top-left coordinate of the bbox.
#         pos = {}
#         for node in subgraph.nodes:
#             bbox = subgraph.nodes[node].get('bbox', (0, 0, 20, 20))
#             pos[node] = (bbox[0], bbox[1])
#
#         # Draw nodes.
#         nx.draw_networkx_nodes(subgraph, pos, node_size=1000, node_color='skyblue', alpha=0.8)
#
#         # Create labels with attributes.
#         labels = {}
#         for node in subgraph.nodes:
#             data = subgraph.nodes[node]
#             if data['node_type'] == 'human':
#                 labels[node] = (f"ID: {data['ID']}\nAge: {data['age']}\n"
#                                 f"Race: {data['race']}\nGender: {data['gender']}\n"
#                                 f"Action: {data['action']}\nSocial Group ID: {data['social_group_id']}\n"
#                                 f"Group Size: {data['group_size']}\nOcclusion: {data['occlusion']}")
#             else:  # object node
#                 labels[node] = f"ID: {data['ID']}\nCategory: {data['category']}"
#
#         nx.draw_networkx_labels(subgraph, pos, labels=labels, font_size=8, font_weight='bold')
#
#         # Draw edges.
#         nx.draw_networkx_edges(subgraph, pos, edgelist=subgraph.edges(), edge_color='gray', width=2, alpha=0.5)
#
#         # Draw edge labels for relationship types.
#         edge_labels = {(u, v): self.graph[u][v]['relation_type'] for u, v in subgraph.edges()}
#         nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_size=8)
#
#         plt.title(f"Frame {frame_id} Visualization")
#         plt.axis('off')
#         plt.show()


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
        # print(frame_id)
        if frame_count ==500:
            break
        # frame_id ='000885.jpg'
        # frame_id = random.choice(list(data_h['labels'].keys()))
        # Extract the number (assumes it's the numeric part before .jpg)
        number = int(frame_id.split('.')[0])
        if number ==536:
           print('stop')

        # ### set robot node
        # scene_graph.add_object(frame_id=frame_id, obj_name='robot', tracking_id='robot',
        #                        category_id='robot', bbox=None)
        # # Check if divisible by 15
        # if number % 15 == 0:
        # # if number % 15 == 0:
        # #     chunk_base = number
        # # else:
        # #     chunk_base = number - (number % 15)
        # #
        # # if number == chunk_base:
        #     ### set object nodes
        #     list_objects_all = Extract_objects(data_o, coco, frame_id)
        #     ### 3D object information
        #     extracted_point_clouds_info = Pointcloud_info_Objects(seq, frame_id, pcd_file, bin_file)
        # for object in list_objects_all:
        #     scene_graph.add_object(frame_id=frame_id, obj_name=object['obj_name'], tracking_id=object['tracking_id'],
        #                                category_id=object['category_id'], bbox=object['obj_bbox'])

        # # Add geometrical OOG (only distance) edges
        # for i in range(len(list_objects_all)):
            # for j in range(i + 1, len(list_objects_all)):

            #     Id_tracki = list_objects_all[i]['tracking_id']
            #     Id_categoryi = list_objects_all[i]['category_id']

            #     Id_trackj = list_objects_all[j]['tracking_id']
            #     Id_categoryj = list_objects_all[j]['category_id']

            #     pointi = pointj = None

            #     for item in extracted_point_clouds_info:
            #         if item.get('tracking_id') == Id_tracki and item.get('category_id') == Id_categoryi:
            #             pointi = item.get('current_point_cloud')
            #         if item.get('tracking_id') == Id_trackj and item.get('category_id') == Id_categoryj:
            #             pointj = item.get('current_point_cloud')

            #     if pointi is not None and pointj is not None:
            #         distance = nearest_distance_kdtree_OO(pointi, pointj)

            #         scene_graph.add_geometric_relationship(
            #             frame_id=frame_id,
            #             node_key_1=f"o_{Id_tracki}_{Id_categoryi}",
            #             node_key_2=f"o_{Id_trackj}_{Id_categoryj}",
            #             relation_type=distance
            #         )

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

            # # Add H-robot_G edges (3D)
            # frame_key = frame_id.replace('jpg', 'pcd')
            # if frame_key in data_h_3d['labels'] and i < len(data_h_3d['labels'][frame_key]):
            #     ID_person, SR_Robot_Ref, distance = Extract_H_robot_G(data_h_3d['labels'][frame_key][i])
            #     scene_graph.add_geometric_relationship(
            #         frame_id=frame_id,
            #         node_key_1=f'h_{ID_person}',
            #         node_key_2='o_robot_robot',
            #         relation_type=distance
            #     )
            #     scene_graph.add_geometric_relationship(
            #         frame_id=frame_id,
            #         node_key_1=f'h_{ID_person}',
            #         node_key_2='o_robot_robot',
            #         relation_type=SR_Robot_Ref
            #     )
            # else:
            #     print(f"Skipping robot relationship for frame '{frame_key}' index {i} — no 3D label available.")

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

                    # print(f'{ID_person_pair} located to {geometry_relation} of {ID_person}')
                # scene_graph.add_geometric_relationship(frame_id=frame_id, node_key_1=f'h_{ID_person_pair}',
                #                                            node_key_2=f'h_{ID_person}', relation_type=SR_Person_Ref)
                # scene_graph.add_geometric_relationship(frame_id=frame_id, node_key_1=f'h_{ID_person}',
                #                                            node_key_2=f'h_{ID_person_pair}', relation_type=distance)
            ### ADD HOG
            # for item in extracted_point_clouds_info:
            #     tracking_id = item.get('tracking_id')
            #     category_id = item.get('category_id')
            #     ID_person,SR_Person_Ref, distance = Extract_HOG(data_h_3d['labels'][pcd][i],item)
            #     ### add geometrical edges  point : the relation is like {ID_person_pair} is {SR_Person_Ref} of {ID_person]
            #     scene_graph.add_geometric_relationship(frame_id=frame_id, node_key_1=f'h_{ID_person}',
            #                                            node_key_2=f'o_{tracking_id}_{category_id}', relation_type=SR_Person_Ref)
            #     scene_graph.add_geometric_relationship(frame_id=frame_id, node_key_1=f'h_{ID_person}',
            #                                            node_key_2=f'h_{f"o_{tracking_id}_{category_id}"}', relation_type=distance)
            #
            #     # scene_graph.visualize_frame(frame_id)


            # # Optionally add object nodes to the frame (e.g., cars, chairs)
            # scene_graph.add_object(frame_id=frame_id, category='car', object_id_in_frame=str(i),
            #                        bbox=(300, 350, 330, 380))  # Adjust the bbox as needed
            # scene_graph.add_object(frame_id=frame_id, category='chair', object_id_in_frame=str(i + 1))
            #
            # Add relationships between humans and objects or between humans themselves




        # scene_graph.visualize_frame(frame_id)
            # if i < len(data_h['labels'][frame_id]) - 1:
            #     scene_graph.add_relationship(frame_id=frame_id, node_key_1=f'h_{i}', node_key_2=f'h_{i + 1}',
            #                                  relation_type='next_to')

        # Visualize a frame (e.g., frame 0)
        # scene_graph.visualize_frame(frame_id)
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
