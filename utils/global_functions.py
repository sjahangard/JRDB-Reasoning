import numpy as np
import open3d as o3d
import json
import random
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.distance import directed_hausdorff
import math
import cv2
import math
import itertools
from Category import all_human_pose_action,OBJECTS,HUMAN_HUMAN_INTERACTION, HUMAN_OBJECT_INTERACTION, GEOMETRY, SOCIAL_AIMS, SALIENT_OBJECTS, BODY_CONTENTS, GENDERS, AGES, RACES, ACTIONS,all_human_pose_action_postfix
from Category import age_corresponding, object_corresponding,object_specific_corresponding,class_thing_id,all_class_id
from Category import *
from global_functions import *
from collections import defaultdict
from pluralizer import Pluralizer
import inflect
from pycocotools.coco import COCO
import numpy as np
import hashlib
import itertools
import json
import argparse
from typing import Any, Dict, List, Tuple, Union
# Argument parser to manage user input

# python your_script.py --set train --task_type VG --spatial 2 --temporal 2 --target_Q human --modality_type video
import copy
import random

def generate_options_count(correct_answer, range_min=1, range_max=12, num_options=5):
    """Generate a list of options including the correct answer and other random values."""
    options = {correct_answer}
    while len(options) < num_options:
        options.add(random.randint(range_min, range_max))
    return sorted(list(options))

def update_single_sample(sample):
    updated_sample = copy.deepcopy(sample)

    all_final_answers = []
    all_target_answers = {}

    for slot_key, slot_data in updated_sample.get("labels", {}).items():
        if "final_bbox" in slot_data:
            final_count = len(slot_data["final_bbox"])
            slot_data["final_answer"] = final_count
            all_final_answers.append(final_count)

        for key, value in slot_data.items():
            if isinstance(value, dict) and "target_bbox" in value:
                count = len(value["target_bbox"])
                value["target_answer"] = count
                if key not in all_target_answers:
                    all_target_answers[key] = []
                all_target_answers[key].append(count)

    q_section = updated_sample.get("question", {})

    if "question" in q_section:
        q_section["question"] = q_section["question"].replace("Find", "Count", 1)

    if all_final_answers:
        main_answer = max(set(all_final_answers), key=all_final_answers.count)
        q_section["options"] = generate_options_count(main_answer)

    if "sub_questions" in q_section:
        updated_sub_questions = {}
        for key, question_text in q_section["sub_questions"].items():
            modified_text = question_text.replace("Find", "Count", 1)
            answers = all_target_answers.get(key, [])
            most_common = max(set(answers), key=answers.count) if answers else 0
            updated_sub_questions[key] = {
                "question": modified_text,
                "options": generate_options_count(most_common)
            }
        q_section["sub_questions"] = updated_sub_questions

    return updated_sample

# def update_single_sample_VQA_Wh(data):
#     """
#     Enrich the question section of a VQA sample JSON with:
#     1. 'options' field from labels['slotX']['final_answer']
#     2. sub_questions with options extracted from each target_answer
#     """
#
#     if "labels" not in data or "question" not in data:
#         raise ValueError("Missing 'labels' or 'question' in input JSON.")
#
#     # Step 1: Find all slot keys in labels
#     for slot_key, slot_data in data["labels"].items():
#         # 1. Add overall options from 'final_answer'
#         if "final_answer" in slot_data:
#             data["question"]["options"] = sorted(list(set(slot_data["final_answer"])))
#
#         # 2. Build sub_questions with options
#         sub_questions = {}
#         for key in slot_data:
#             if key not in {"final_answer", "final_ID", "image_ids"}:
#                 if isinstance(slot_data[key], dict) and "target_answer" in slot_data[key]:
#                     question_text = f"What are the {key} of the persons who are located at moderate distance and back relative to me(robot)?"
#                     options = sorted(list(set(slot_data[key]["target_answer"])))
#                     sub_questions[key] = {
#                         "question": question_text,
#                         "options": options
#                     }
#
#         data["question"]["sub_questions"] = sub_questions
#
#     return data

# def update_single_sample_VQA_Wh(data):
#     """
#     Enrich the question section of a VQA sample JSON with:
#     1. 'sub_questions' — use existing question text and just add predefined options.
#     2. Set top-level 'options' in question as union of all sub_question options.
#     Supports compound keys like "race, action".
#     """
#
#     if "labels" not in data or "question" not in data:
#         raise ValueError("Missing 'labels' or 'question' in input JSON.")
#
#     # Predefined options for each key
#     # Options = {
#     #     'age': ['child', 'adolescent', 'young', 'middle-aged', 'elderly'],
#     #     'race': ['White', 'Black', 'Asian', 'others'],
#     #     'gender': ['male', 'female'],
#     #     'action_labels': all_human_pose_action,
#     #     'action': all_human_pose_action,
#     #     'h_dis_robot': ['very close', 'close', 'moderate', 'far', 'very far'],
#     #     'hhi': HUMAN_HUMAN_INTERACTION,
#     #     'hhG': GEOMETRY,
#     #     'hoi': ['holding cup', 'carrying bag', 'walking on the floor'],
#     #     'aim': SOCIAL_AIMS,
#     #     'category': OBJECTS,
#     #     'obj_dis_robot': ['very close', 'close', 'moderate', 'far', 'very far'],
#     #     'SR_robot_ref': GEOMETRY,
#     #     # Compound keys should have defined or dynamically merged options
#     #     'race, action': list(set(['White', 'Black', 'Asian', 'others']) | set(['sitting', 'standing', 'walking'])),
#     # }
#
#     sub_questions = data["question"].get("sub_questions", {})
#     all_options = set()
#
#     for slot_key, slot_data in data["labels"].items():
#         for key in slot_data:
#             if key not in {"final_answer", "final_ID", "image_ids"}:
#                 value = slot_data[key]
#                 if isinstance(value, dict) and "target_answer" in value:
#                     individual_keys = [k.strip() for k in key.split(",")]
#
#                     for sub_key in individual_keys:
#                         if sub_key not in Options:
#                             raise ValueError(f"Missing predefined options for key: {sub_key}")
#                         options = sorted(Options[sub_key])
#
#                         if sub_key in sub_questions:
#                             existing_text = sub_questions[sub_key]
#                             sub_questions[sub_key] = {
#                                 "question": existing_text if isinstance(existing_text, str) else existing_text.get("question", ""),
#                                 "options": options
#                             }
#                             all_options.update(options)
#
#                     # Handle compound keys like 'race, action'
#                     if key in Options and key in sub_questions:
#                         options = sorted(Options[key])
#                         sub_questions[key] = {
#                             "question": sub_questions[key] if isinstance(sub_questions[key], str) else sub_questions[key].get("question", ""),
#                             "options": options
#                         }
#                         all_options.update(options)
#
#     # Set top-level question options after sub_questions are processed
#     data["question"]["sub_questions"] = sub_questions
#     data["question"]["options"] = sorted(all_options)
#
#     return data

# def update_single_sample_VQA_Wh(data):
#     """
#     Enrich the question section of a VQA sample JSON with:
#     1. 'sub_questions' — use existing question text and add predefined options (max 15 items).
#     2. Set top-level 'options' in question as union of all sub_question options (max 15 items).
#     Supports compound keys like "race, action".
#     """
#
#     if "labels" not in data or "question" not in data:
#         raise ValueError("Missing 'labels' or 'question' in input JSON.")
#
#     max_options = 15
#
#     # Options = {
#     #     'age': ['child', 'adolescent', 'young', 'middle-aged', 'elderly'],
#     #     'race': ['White', 'Black', 'Asian', 'others'],
#     #     'gender': ['male', 'female'],
#     #     'action': ['sitting', 'standing', 'walking', 'running', 'jumping', 'laying', 'kneeling', 'climbing', 'dancing', 'crouching', 'stretching', 'waving', 'bending', 'squatting', 'hopping'],
#     #     'h_dis_robot': ['very close', 'close', 'moderate', 'far', 'very far'],
#     #     'obj_dis_robot': ['very close', 'close', 'moderate', 'far', 'very far'],
#     #     'hoi': ['holding cup', 'carrying bag', 'walking on the floor'],
#     #     'SR_robot_ref': ['left', 'right', 'front', 'behind'],
#     #     # Add other keys as needed
#     # }
#
#     sub_questions = data["question"].get("sub_questions", {})
#     all_options = set()
#
#     for slot_key, slot_data in data["labels"].items():
#         for key in slot_data:
#             if key not in {"final_answer", "final_ID", "image_ids"}:
#                 value = slot_data[key]
#                 if isinstance(value, dict) and "target_answer" in value:
#                     individual_keys = [k.strip() for k in key.split(",")]
#
#                     # Handle individual keys
#                     for sub_key in individual_keys:
#                         if sub_key not in Options:
#                             raise ValueError(f"Missing predefined options for key: {sub_key}")
#                         options = sorted(Options[sub_key])[:max_options]
#
#                         if sub_key in sub_questions:
#                             existing_text = sub_questions[sub_key]
#                             sub_questions[sub_key] = {
#                                 "question": existing_text if isinstance(existing_text, str) else existing_text.get("question", ""),
#                                 "options": options
#                             }
#                             all_options.update(options)
#
#                     # Handle compound key
#                     if "," in key and key in sub_questions:
#                         combined_options = set()
#                         for sub_key in individual_keys:
#                             if sub_key not in Options:
#                                 raise ValueError(f"Missing predefined options for key: {sub_key}")
#                             combined_options.update(Options[sub_key])
#                         combined_options = sorted(combined_options)[:max_options]
#
#                         existing_text = sub_questions[key]
#                         sub_questions[key] = {
#                             "question": existing_text if isinstance(existing_text, str) else existing_text.get("question", ""),
#                             "options": combined_options
#                         }
#                         all_options.update(combined_options)
#
#     # Trim top-level options
#     data["question"]["sub_questions"] = sub_questions
#     data["question"]["options"] = sorted(all_options)[:max_options]
#
#     return data

import random

def update_single_sample_VQA_Wh(data):
    """
    Enrich the question section of a VQA sample JSON with:
    1. 'sub_questions' — use existing question text and add randomly permuted options (max 15).
    2. Set top-level 'options' in question as union of all sub_question options (max 15).
    Supports compound keys like "race, action".
    """

    if "labels" not in data or "question" not in data:
        raise ValueError("Missing 'labels' or 'question' in input JSON.")

    max_options = 15

    # Options = {
    #     'age': ['child', 'adolescent', 'young', 'middle-aged', 'elderly'],
    #     'race': ['White', 'Black', 'Asian', 'others'],
    #     'gender': ['male', 'female'],
    #     'action': ['sitting', 'standing', 'walking', 'running', 'jumping', 'laying', 'kneeling',
    #                'climbing', 'dancing', 'crouching', 'stretching', 'waving', 'bending', 'squatting', 'hopping'],
    #     'h_dis_robot': ['very close', 'close', 'moderate', 'far', 'very far'],
    #     'obj_dis_robot': ['very close', 'close', 'moderate', 'far', 'very far'],
    #     'hoi': ['holding cup', 'carrying bag', 'walking on the floor'],
    #     'SR_robot_ref': ['left', 'right', 'front', 'behind'],
    #     # Add other keys as needed
    # }

    sub_questions = data["question"].get("sub_questions", {})
    all_options = set()

    for slot_key, slot_data in data["labels"].items():
        for key in slot_data:
            if key not in {"final_answer", "final_ID", "image_ids"}:
                value = slot_data[key]
                if isinstance(value, dict) and "target_answer" in value:
                    individual_keys = [k.strip() for k in key.split(",")]

                    # Handle individual sub-keys
                    for sub_key in individual_keys:
                        if sub_key not in Options:
                            raise ValueError(f"Missing predefined options for key: {sub_key}")
                        permuted_options = random.sample(Options[sub_key], min(len(Options[sub_key]), max_options))

                        if sub_key in sub_questions:
                            existing_text = sub_questions[sub_key]
                            sub_questions[sub_key] = {
                                "question": existing_text if isinstance(existing_text, str) else existing_text.get("question", ""),
                                "options": permuted_options
                            }
                            all_options.update(permuted_options)

                    # Handle compound key
                    if "," in key and key in sub_questions:
                        combined = set()
                        for sub_key in individual_keys:
                            if sub_key not in Options:
                                raise ValueError(f"Missing predefined options for key: {sub_key}")
                            combined.update(Options[sub_key])
                        combined = list(combined)
                        permuted_combined = random.sample(combined, min(len(combined), max_options))

                        existing_text = sub_questions[key]
                        sub_questions[key] = {
                            "question": existing_text if isinstance(existing_text, str) else existing_text.get("question", ""),
                            "options": permuted_combined
                        }
                        all_options.update(permuted_combined)

    # Set final question options (also permuted and limited)
    final_all_options = list(all_options)
    data["question"]["sub_questions"] = sub_questions
    data["question"]["options"] = random.sample(final_all_options, min(len(final_all_options), max_options))

    return data


def update_data_structure_all_slots(samples,type_VQA):
    if type_VQA == "Count":
       return [update_single_sample(sample) for sample in samples]
    elif type_VQA == "Wh":
        return [update_single_sample_VQA_Wh(sample) for sample in samples]


def choose_question_type():
    """Returns 'Count' with 1/3 probability, 'Wh' with 2/3 probability."""
    return "Count" if random.randint(1, 3) == 1 else "Wh"
def parse_arguments():
    parser = argparse.ArgumentParser(description="Parser for different settings")

    # Set
    parser.add_argument('--set', choices=['train', 'test'], default='test', help="Set to use (train/test)")

    # Task Type
    parser.add_argument('--task_type', choices=['VG', 'VQA'], default='VG', help="Type of task ('VG' or 'VQA')")

    # Spatial
    parser.add_argument('--spatial', choices=['1', '2', '3'], default='2', help="Spatial setting ('1', '2', '3')")

    # Temporal
    parser.add_argument('--temporal', choices=[1, 2, 3], type=int, default=2, help="Temporal setting (1, 2, 3)")

    # Target Question
    parser.add_argument('--target_Q', choices=['human', 'human&object', 'object'], default='human', help="Target Question type ('human', 'human&object', 'object')")

    # Modality Type
    parser.add_argument('--modality_type', choices=['image', 'video'], default='video', help="Modality type ('image' or 'video')")

    return parser.parse_args()
def read_pcd(rest_path):
    pcd_file = "/Users/sjah0003/PycharmProjects/pythonProject_visulizastion/JRDB2022/siminpanoptic_3d_full/pointcloud/train_dataset_with_activity/merged_velodyne/"
    point_cloud = o3d.io.read_point_cloud(pcd_file + rest_path)
    return np.asarray(point_cloud.points)
def read_bin(rest_path):
    bin_file = "/Users/sjah0003/PycharmProjects/pythonProject_visulizastion/JRDB2022/siminpanoptic_3d_full/labels/train_dataset_with_activity/"
    label = np.fromfile(bin_file + rest_path, dtype=np.uint32)
    track_ids = label >> 16
    class_ids = label & 0xFFFF
    return track_ids, class_ids
def read_json_file(path_label, path_image, seq1):
    # for filename in os.listdir(path):
    print(seq1)
    filename = seq1
    f = open(path_label + filename)
    data = json.load(f)


    return data
def int_to_file_names(num):
    # Convert integer to a zero-padded string with 6 digits
    padded_num = str(num).zfill(6)
    pcd_filename = f"{padded_num}.pcd"
    bin_filename = f"{padded_num}.bin"
    return pcd_filename, bin_filename
def return_group_size(data,image,person):
    group_size='impossible'
    group_size = len([others for others in data['labels'][image] if
                      person["social_group"]['cluster_ID'] == others["social_group"]['cluster_ID']])
    return str(group_size)
def extract_person_attributes(person):
    """
    Extract demographic and action attributes from a person dictionary.

    Returns a tuple: (age, race, gender, action, social_group, bbox)
    If certain attributes cannot be determined, they default to 'impossible'.
    """
    # Default values for all attributes
    age = race = gender = action = social_group = bbox = None
    social_group_id = group_size = human_id = None
    occlusion = None

    attributes = person.get('attributes', {})
    occlusion = attributes.get('occlusion')
    area = attributes.get('area', 0)

    demographics_list = person.get("demographics_info", [])
    if demographics_list:
        demographics_info = demographics_list[0]
        bbox = person.get('box', None)
        human_id = person.get('label_id', None)

        # Extract demographic details if available
        gender_info = demographics_info.get('gender', {})
        age_info = demographics_info.get('age', {})
        race_info = demographics_info.get('race', {})

        if gender_info:
            gender = list(gender_info.keys())[0].lower()
        if age_info:
            age_key = list(age_info.keys())[0]
            # age_corresponding is assumed to be a pre-defined mapping for age values
            age = age_corresponding.get(age_key, None)
        if race_info:
            race_key = list(race_info.keys())[0]
            # Process race only if it is not 'impossible' or 'Others'
            if race_key not in [None, 'Others']:
                parts = race_key.split('/')
                race = parts[1] if len(parts) > 1 else race_key
            else:
                race = None

    # Process action labels if available
    action_label_info = person.get("action_label", {})
    for act in action_label_info.keys():
        if act in all_human_pose_action:
            # Normalize certain action labels
            if act == 'going downstairs':
                action = 'going down'
            elif act == 'going upstairs':
                action = 'going up'
            else:
                action = act
            break

    # Additional logic for social group can be added here
    social_group_id = str(person.get('social_group', {}).get('cluster_ID', None))
    group_size = person.get('group_size', None)

    if gender == "impossible": gender = None
    if age == "impossible": age = None
    if race == "impossible": race = None
    if action == "impossible": action = None

    return gender, age, race, action, social_group_id, group_size, bbox, human_id, occlusion
def Extract_HHI_HOI(person):
    ID_person = person.get("label_id", {})

    HHI_label_info = person.get("HHI", {})
    HOI_label_info = person.get("HOI", {})

    pair_dic={}
    # Check if there are any HHI entries
    if len(HHI_label_info) > 0:
        for interaction in HHI_label_info:
            hhi_R_inter, pair_person = (interaction['inter_labels'], interaction['pair'])
            if pair_person not in pair_dic:
                pair_dic[pair_person]=[]
            # Extract attributes of the pair
            # pair_gender_R = pair_age_R = pair_race_R = pair_box_R = None
            # # Append the result to the list
            if list(hhi_R_inter.keys())[0] not in pair_dic[pair_person]:
               pair_dic[pair_person].append(list(hhi_R_inter.keys())[0])


    # print("HHI_label_info",HHI_label_info)
    HHI_labels = [list(item['inter_labels'].keys())[0] for item in HHI_label_info]

    # print("HOI_label_info", HOI_label_info)
    HOI_labels = [list(item['inter_labels'].keys())[0] for item in HOI_label_info]

    return ID_person,HHI_label_info,HOI_label_info
def compute_relative_coordinates(bbox_ref,bbox):
    """
     Compute the coordinates of a point relative to a bounding box.

     Args:
         point (list or tuple): The global coordinates of the point [x, y, z].
         bbox (dict): The bounding box parameters with keys:
                      'cx', 'cy', 'cz' (center of the box),
                      'rot_z' (rotation angle in radians).

     Returns:
         list: The coordinates of the point relative to the bounding box [x_local, y_local, z_local].
     """
    # Extract bounding box parameters
    cx_ref, cy_ref, cz_ref = bbox_ref['cx'], bbox_ref['cy'], bbox_ref['cz']
    rot_z_ref = bbox_ref['rot_z']

    # cx, cy, cz = bbox['cx'], bbox['cy'], bbox['cz']
    cx, cy, cz = bbox
    # Translate the point to the bounding box center
    translated_point_ref = [ cx_ref-cx, cy_ref-cy,  cz_ref-cz]

    # Create the rotation matrix for rotation around the z-axis
    rotation_matrix = np.array([
        [np.cos(rot_z_ref), -np.sin(rot_z_ref), 0],
        [np.sin(rot_z_ref), np.cos(rot_z_ref), 0],
        [0, 0, 1]
    ])

    # Rotate the translated point
    rotated_point = np.dot(rotation_matrix, translated_point_ref)

    return rotated_point.tolist()
def Position_relative_to_Ref(B1,B2):
    # Define the vectors B1, B2, and Vego
    # B1 =[x,y]
    # B1 = np.array([-1, 1])
    # B2 = np.array([1, 1])
    Vego = np.array([1, 0])
    Z=B1[2]-B2[2]
    distance_Z=abs(Z)
    # Compute the signed angle using atan2
    theta_signed_rad = np.arctan2(B1[1] - B2[1], B1[0] - B2[0]) - np.arctan2(Vego[1], Vego[0])

    # Convert the signed angle to degrees
    theta_signed_deg = np.degrees(theta_signed_rad)

    if distance_Z>1.5:
        # Determine the relation based on the angle
        if Z>0:
           Z_dimention = 'up'
        else:
           Z_dimention = 'down'
        if -30 <= theta_signed_deg <= 30:
            relation = f'{Z_dimention} front'
        elif 30 < theta_signed_deg <= 45:
            relation = f'{Z_dimention} front left'
        elif 45 < theta_signed_deg <= 135:
            relation = f'{Z_dimention} left'
        elif 135 < theta_signed_deg <= 145:
            relation = f'{Z_dimention} back left'
        elif -30 >= theta_signed_deg > -45:
            relation = f'{Z_dimention} front right'
        elif -45 >= theta_signed_deg > -135:
            relation = f'{Z_dimention} right'
        elif -135 >= theta_signed_deg > -145:
            relation = f'{Z_dimention} back right'
        else:
            relation = f'{Z_dimention} back'
    else:
        # Determine the relation based on the angle
        if -30 <= theta_signed_deg <= 30:
            relation = 'front'
        elif 30 < theta_signed_deg <= 45:
            relation = 'front left'
        elif 45 < theta_signed_deg <= 135:
            relation = 'left'
        elif 135 < theta_signed_deg <= 145:
            relation = 'back left'
        elif -30 >= theta_signed_deg > -45:
            relation = 'front right'
        elif -45 >= theta_signed_deg > -135:
            relation = 'right'
        elif -135 >= theta_signed_deg > -145:
            relation = 'back right'
        else:
            relation = 'back'
    return relation
def draw_bounding_box(binary_mask):
    # Ensure the mask is in binary format (0 and 1)
    binary_mask = (binary_mask > 0).astype(np.uint8)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the bounding box coordinates around the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Create a copy of the mask to draw the bounding box
        mask_with_box = cv2.cvtColor(binary_mask * 255, cv2.COLOR_GRAY2BGR)

        # Draw the bounding box on the image
        # cv2.rectangle(mask_with_box, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return mask_with_box, [x, y, w, h]
    else:
        return None, None
def nearest_distance_kdtree(per3_D, extracted_point_clouds_object, k=10):
    # Extract the 3D coordinates of the bounding box center
    points_i = np.array([per3_D['box']['cx'], per3_D['box']['cy'], per3_D['box']['cz']])

    # Ensure extracted point cloud is a valid ndarray
    points_j = np.array(extracted_point_clouds_object)

    if points_j.size == 0:
        return []

    # Build KDTree for the points
    tree_j = cKDTree(points_j)

    # Find the nearest neighbors from human bbox center to the point cloud
    k_neighbors = min(k, len(points_j))
    distances, _ = tree_j.query(points_i, k=k_neighbors, workers=-1)

    # Compute and return the mean distance to the nearest neighbors
    mean_dist_i_to_j = np.mean(distances)

    return mean_dist_i_to_j

def read_pcd(pcd_file):
    point_cloud = o3d.io.read_point_cloud(pcd_file)
    return np.asarray(point_cloud.points)
def read_bin(bin_file):
    label = np.fromfile(bin_file, dtype=np.uint32)
    track_ids = label >> 16
    class_ids = label & 0xFFFF
    return track_ids, class_ids
def Extract_H_robot_G(frame_persons, human_id):
    SR_Robot_Ref = None
    distance = None

    for person in frame_persons:
        ID_person = person.get("label_id", {})
        if ID_person == human_id:
            distance = person['attributes']['distance']
            SR_Robot_Ref = Position_relative_to_Ref(
                [person['box']['cx'], person['box']['cy'], person['box']['cz']],
                [0, 0, 0]
            )
            break
    distance1 = discretize_distances2([distance])

    return SR_Robot_Ref, distance1, person['box']
def Extract_objects(data_o,coco,image):
    class_list = set()
    # frame_number=int(image.split('.')[0])
    # inx=int(frame_number%15)
    # img_ids = data_o["images"][inx]
    # coco = COCO(annotation_dir)
    img_ids = data_o["images"]
    img_ids_dict = {}
    for img_id in img_ids:
        img_ids_dict[img_id["id"]] = img_id["file_name"]
    # print(img_ids_dict.keys())
    annos_ids = coco.getAnnIds(img_ids_dict.keys())
    annotations = coco.loadAnns(annos_ids)
    # list of annotations to dictf
    annos_dict = defaultdict(list)

    error_cnt = 0
    # Preprocessing step: map category IDs to names for faster lookup
    # name2isthing = {cat["name"]: cat["isthing"] for cat in coco.loadCats(coco.getCatIds())}
    cat_id_to_name = {cat["id"]: cat["name"] for cat in coco.loadCats(coco.getCatIds())}
    for anno in annotations:
        cat_name = cat_id_to_name[anno["category_id"]]
        # if not show_stitch:  # means that unprocessed data
        #     if cat_name == 't_misc':
        #         cat_name = "m_" + anno['attributes']['description']
        #     # lower-case
        #     cat_name = cat_name.lower()
        anno["name"] = cat_name

        # append class name to class_list
        class_list.add(cat_name)
        # print(class_list)
        # print("new_classes", class_list - set(class_order))
        # Assign color to each annotation
        color = class_colors[cat_name]
        anno['color'] = color
        # if name2isthing[cat_name]:
        #     try:
        #         if 'tracking_id' in anno['attributes']:
        #             id = int(anno['attributes']['tracking_id'])
        #         elif 'floor_id' in anno['attributes']:
        #             id = int(anno['attributes']['floor_id'])
        #         else:
        #             raise Exception("No tracking_id or floor_id in attributes")
        #         # generate a unique color using combination of id and cat_name
        #         # anno['instance_color'] = palette[(anno["category_id"] ** 2 * id ** 3) % MAX_OBJECT].tolist()
        #         anno['instance_color'] = get_color_from_id_and_name(id, cat_name)
        #
        #     except:
        #         # handle_no_tracking_id_error(anno, annotation_dir, location)
        #         # print('OPs')
        #         id = 0
        #         anno['attributes']['tracking_id'] = -1
        annos_dict[anno["image_id"]].append(anno)
    img_id = int(image.split('.')[0]) // 15 + 1
    # print(img_id)
    annos = annos_dict[img_id]
    list_objects_all = []
    # get segmentation
    # semantic_mask = np.zeros((3760, 480, 3), dtype=np.uint8)
    # instance_mask = np.zeros((3760, 480, 3), dtype=np.uint8)
    # panoptic_mask = np.zeros((3760, 480, 3), dtype=np.uint8)
    # binary_semantic_mask = np.zeros((3760, 480), dtype=np.uint8)
    # binary_instance_mask = np.zeros((3760, 480), dtype=np.uint8)
    # binary_panoptic_mask = np.zeros((3760, 480), dtype=np.uint8)
    bbs_o = []
    mask = None
    for anno in annos:
        # if anno['name'] == 'clock':
        #    # print(anno['name'])
        # if anno['name'] not in list_objects_all and anno['name'] in object_specific_corresponding:
        #     list_objects_all.append((anno['name'],(anno['tracking_id'],anno['category_id'])))
        if anno['category_id']!=63:

        # if anno['name'] == object_Q[0] and anno['tracking_id'] == object_Q[1][0] and anno['category_id'] == object_Q[1][
        #     1]:
            mask = coco.annToMask(anno)  # mask  0-1 of the object
            mask, bb_o = draw_bounding_box(mask)
            list_objects_all.append({'obj_name':anno['name'],'tracking_id':str(anno['tracking_id']), 'category_id':str(anno['category_id']),'obj_bbox':bb_o})
            # bbs_o.append((bb_o, (anno['tracking_id'], anno['category_id'])))
            # if self.Object_detected_flag==1 and anno['tracking_id']==self.current_obj_track_ID and self.current_obj_category_ID == anno['category_id'] :
            #     mask = self.coco.annToMask(anno) * 255
            #     red_overlay = np.zeros_like(self.Current_img)
            #     red_overlay[:, :, 2] = 255  # Set red channel to 255 (full intensity)
            #     # Apply the mask to the red overlay
            #     red_masked_overlay = cv2.bitwise_and(red_overlay, red_overlay, mask=mask)
            #     # Combine the original image with the red masked overlay
            #     self.Current_img = cv2.addWeighted(self.Current_img, 1, red_masked_overlay, 0.5, 0)
    return list_objects_all
def is_number_in_dict(number, dictionary):
    return number in dictionary.values()
def calculate_center_of_point_cloud(pcd):
    return np.mean(pcd, axis=0)
def calculate_bounding_box_of_point_cloud(pcd):
    # Check if the point cloud array is empty
    if pcd.size == 0:
        raise ValueError("The point cloud array is empty. Cannot calculate bounding box.")
        return None, None, None

    min_corner = np.min(pcd, axis=0)
    max_corner = np.max(pcd, axis=0)
    center = (min_corner + max_corner) / 2
    return min_corner, max_corner, center
def filter_points(class_ids, class_thing_id, pcd_info, track_ids):
    # Initialize minimum points count and result lists
    min_points = 10000
    list_pcd, list_track_id, list_class_id = [], [], []

    # Filter points that correspond to "things"
    for inx, label in enumerate(class_ids):
        # if is_number_in_dict(label.item(), class_thing_id):
            list_pcd.append(pcd_info[inx])
            list_track_id.append(track_ids[inx])
            list_class_id.append(class_ids[inx])

    # Find all unique combinations of track IDs and label IDs
    unique_combinations = set(zip(list_track_id, list_class_id))

    # Initialize dictionaries to store results
    extracted_point_clouds = []
    object_centers = []
    object_bounding_boxes = []

    # Process each unique combination
    for track_id, label_id in unique_combinations:
        current_point_cloud = []
        for inx in range(len(list_track_id)):
            if list_track_id[inx] == track_id and list_class_id[inx] == label_id:
                current_point_cloud.append(list_pcd[inx])

        if len(current_point_cloud) > 0:
            # Convert the list of points to a numpy array
            current_point_cloud = np.array(current_point_cloud)
            if current_point_cloud.shape[0] < min_points:
                min_points = current_point_cloud.shape[0]

            # Store the point cloud directly
            extracted_point_clouds.append( {'tracking_id':str(track_id), 'category_id': str(label_id), 'current_point_cloud':current_point_cloud,'center_of_point_cloud':calculate_center_of_point_cloud(current_point_cloud),'bounding_box_of_point_cloud':calculate_bounding_box_of_point_cloud(current_point_cloud)})
            # object_centers.append( {'tracking_id':str(track_id), 'category_id': str(label_id), 'center_of_point_cloud':calculate_center_of_point_cloud(current_point_cloud)})
            # object_bounding_boxes.append( {'tracking_id':str(track_id), 'category_id': str(label_id),'bounding_box_of_point_cloud':calculate_bounding_box_of_point_cloud(current_point_cloud)})

    return extracted_point_clouds
def Pointcloud_info_Objects(seq_name,image,pcd_file,bin_file):
    pcd_filename, bin_filename = int_to_file_names(int(image.split('.')[0]))
    pcd_info = read_pcd(pcd_file + seq_name + '/' + pcd_filename)
    track_ids, class_ids = read_bin(bin_file + seq_name + '/' + bin_filename)
    extracted_point_clouds, object_centers, object_bounding_boxes = filter_points(class_ids, class_thing_id,
                                                                                  pcd_info, track_ids)
    return extracted_point_clouds, object_centers, object_bounding_boxes
def nearest_distance_kdtree_OO(source_points, target_points, k=10):
    """
    Compute the mean distance from each point in source_points to its k nearest neighbors in target_points.

    Parameters:
        source_points (array-like): Array of shape (n, 3), representing the source point cloud.
        target_points (array-like): Array of shape (m, 3), representing the target point cloud.
        k (int): Number of nearest neighbors to consider.

    Returns:
        float: Mean distance from source_points to target_points using k-nearest neighbors.
               Returns np.nan if target_points is empty.
    """
    source_points = np.asarray(source_points)
    target_points = np.asarray(target_points)

    if target_points.size == 0 or source_points.size == 0:
        return np.nan

    k_neighbors = min(k, len(target_points))
    tree = cKDTree(target_points)
    distances, _ = tree.query(source_points, k=k_neighbors, workers=-1)

    # Handle the case when k=1 and output is 1D
    distances = np.atleast_2d(distances)

    mean_dist = np.mean(distances)
    return round(mean_dist, 3)

def Pointcloud_info_Objects(seq_name,image,pcd_file,bin_file):
    pcd_filename, bin_filename = int_to_file_names(int(image.split('.')[0]))
    pcd_info = read_pcd(pcd_file + seq_name + '/' + pcd_filename)
    track_ids, class_ids = read_bin(bin_file + seq_name + '/' + bin_filename)
    extracted_point_clouds, object_centers, object_bounding_boxes = filter_points(class_ids, class_thing_id,
                                                                                  pcd_info, track_ids)
    return extracted_point_clouds, object_centers, object_bounding_boxes
def Extract_HHG(person, person_pair):
    # Preserve input extraction as-is
    ID_person = person.get("label_id", {})
    ID_person_pair = person_pair.get("label_id", {})
    # if ID_person == 'pedestrian:27' and ID_person_pair =='pedestrian:29':
    #     print('stop')

    per3_D_box = person.get('box', {})
    other_per_box = person_pair.get('box', {})

    # per3_D_box= {'cx': 0.596, 'cy': 2.817, 'cz': -0.235}
    # other_per_box ={'cx': -0.202, 'cy': 4.336, 'cz': -0.183}

    # per3_D_box = {
    #                 "cx": 0.36068,
    #                 "cy": 2.84801,
    #                 "cz": 0.01485,
    #                 "h": 1.8897,
    #                 "l": 1.0,
    #                 "rot_z": 3.50856,
    #                 "w": 0.55
    #             }
    #
    # other_per_box = {
    #                 "cx": -0.24806,
    #                 "cy": 3.41734,
    #                 "cz": -0.0294,
    #                 "h": 1.8012,
    #                 "l": 1.03,
    #                 "rot_z": -2.17123,
    #                 "w": 0.55
    #             }

    # Calculate the spatial geometry
    other_point_translated = compute_relative_coordinates(
        per3_D_box,
        [other_per_box['cx'], other_per_box['cy'], other_per_box['cz']]
    )
    SR_Person_Ref = Position_relative_to_Ref(
        other_point_translated,
        [0, 0, 0]
    )

    # Compute Euclidean distance from reference at origin
    distance = math.sqrt(
        # other_point_translated[0] ** 2 +
        other_point_translated[0] ** 2 +
        other_point_translated[1] ** 2 +
        other_point_translated[2] ** 2
    )

    # Discretize the distance and extract value
    Dis_list = discretize_distances2([distance])
    Dis = Dis_list[0] if isinstance(Dis_list, list) and Dis_list else Dis_list

    # Return IDs and combined relation string
    return ID_person, ID_person_pair, f"{SR_Person_Ref} and {Dis}"

def Extract_HOG(person,item):

    ID_person = person.get("label_id", {})

    per3_D_box = person.get('box', {})

    distance = nearest_distance_kdtree(person, item.get('current_point_cloud'), k=10)

    other_obj_point_translated = compute_relative_coordinates(per3_D_box, item.get('center_of_point_cloud'))
    SR_Person_Ref = Position_relative_to_Ref(
        [other_obj_point_translated[0], other_obj_point_translated[1], other_obj_point_translated[2]], [0, 0, 0])

    return ID_person,SR_Person_Ref, round(distance, 3)

def Pointcloud_info_Objects(seq_name,image,pcd_file,bin_file):
    pcd_filename, bin_filename = int_to_file_names(int(image.split('.')[0]))
    seq_name= seq_name.split('.')[0]
    pcd_info = read_pcd(pcd_file + seq_name + '/' + pcd_filename)
    track_ids, class_ids = read_bin(bin_file + seq_name + '/' + bin_filename)
    extracted_point_clouds = filter_points(class_ids, class_thing_id,pcd_info, track_ids)
    return extracted_point_clouds
from itertools import combinations

# def extract_different_value_combinations_all_keys(data, T=2):
#     result = {}
#     skip_variation_check_keys = {'age', 'race', 'gender'}
#
#     for person_id, attributes in data.items():
#         person_result = {}
#
#         for key, segments in attributes.items():
#             formatted_segments = []
#
#             for seg in segments:
#                 val = seg.get('value')
#                 if val is None:
#                     continue
#                 # Standardize to string for comparison
#                 if isinstance(val, (tuple, list)):
#                     val_str = str(sorted(val))
#                 else:
#                     val_str = str(val)
#                 formatted_segments.append({
#                     'start': seg['start'],
#                     'end': seg['end'],
#                     'variable': [val_str]
#                 })
#
#             if not formatted_segments:
#                 continue
#
#             if key in skip_variation_check_keys:
#                 # Include all single segments as 1-tuples
#                 person_result[key] = [tuple([fs]) for fs in formatted_segments]
#             else:
#                 # Only include T-combinations with distinct values
#                 valid_combos = []
#                 for combo in combinations(formatted_segments, T):
#                     values = [c['variable'][0] for c in combo]
#                     if len(set(values)) == T:
#                         valid_combos.append(combo)
#
#                 if valid_combos:
#                     person_result[key] = valid_combos
#
#         if person_result:
#             result[person_id] = person_result
#
#     return result
from itertools import combinations

# Define which keys carry multiple interaction values with metadata
multi_interaction_keys = {'hhi', 'hhg', 'hoi'}
skip_variation_check_keys = {'age', 'race', 'gender'}

# Mapping from key to the id field used for matching combos
id_field_map = {
    'hhi': 'id_pair',
    'hhg': 'id_pair',
    'hoi': 'id'
}

# Mapping from key to additional metadata fields to carry
metadata_fields_map = {
    'hhi': ['gender_pair', 'age_pair', 'race_pair'],
    'hhg': ['gender_pair', 'age_pair', 'race_pair'],
    'hoi': ['obj_name', 'tracking_id', 'category_id']
}

def extract_different_value_combinations_all_keys(data, T=2):
    result = {}

    for person_id, attributes in data.items():
        person_result = {}

        for key, segments in attributes.items():
            formatted_segments = []

            for seg in segments:
                raw = seg.get('value')
                if raw is None:
                    continue

                # Handle multi-interaction keys: extract each interaction separately
                if key in multi_interaction_keys and isinstance(raw, dict):
                    interactions = raw.get('interactions', [])
                    id_field = id_field_map[key]
                    fields = metadata_fields_map.get(key, [])

                    for inter in interactions:
                        meta = {fld: raw.get(fld) for fld in fields}
                        # include the id for matching combos
                        meta[id_field] = raw.get(id_field)
                        formatted_segments.append({
                            'start': seg['start'],
                            'end': seg['end'],
                            'variable': [str(inter)],
                            **meta
                        })
                    continue

                # Standardize to string for comparison for other keys
                if isinstance(raw, (tuple, list)):
                    val_str = str(sorted(raw))
                else:
                    val_str = str(raw)

                formatted_segments.append({
                    'start': seg['start'],
                    'end': seg['end'],
                    'variable': [val_str]
                })

            if not formatted_segments:
                continue

            # Demographic keys: all singletons
            if key in skip_variation_check_keys:
                person_result[key] = [tuple([fs]) for fs in formatted_segments]
            else:
                # Build T-combinations, enforce distinct variable values
                valid_combos = []
                for combo in combinations(formatted_segments, T):
                    values = [c['variable'][0] for c in combo]
                    if len(set(values)) != T:
                        continue

                    # For multi-interaction keys, enforce same id field across combo
                    if key in multi_interaction_keys:
                        id_field = id_field_map[key]
                        id_vals = {c.get(id_field) for c in combo}
                        if len(id_vals) != 1:
                            continue

                    valid_combos.append(combo)

                if valid_combos:
                    person_result[key] = valid_combos

        if person_result:
            result[person_id] = person_result

    return result




def cal_over_laps(value, T,S):
    # def calculate_overlap(intervals):
    #     """
    #     Calculate the overlap across multiple intervals.
    #     :param intervals: List of interval tuples [(start, end), ...].
    #     :return: Overlapping interval (start, end) or None if no overlap.
    #     """
    #     start = max(interval['start'] for interval in intervals)
    #     end = min(interval['end'] for interval in intervals)
    #     return (start, end) if start <= end else None
    def calculate_overlap(intervals):
        """
        Calculate the overlap across multiple intervals.
        :param intervals: List of dicts or tuples [(start, end), ...].
        :return: Overlapping interval (start, end) or None if no overlap or no intervals.
        """
        # 1) early exit on no intervals
        if not intervals:
            return None

        # 2) if your intervals are dicts with 'start'/'end'
        try:
            start = max(interval['start'] for interval in intervals)
            end = min(interval['end'] for interval in intervals)
        except (KeyError, TypeError):
            # maybe intervals are simple tuples?
            start = max(i[0] for i in intervals)
            end = min(i[1] for i in intervals)

        # 3) only return a real overlap if start ≤ end
        return (start, end) if start <= end else None

    labels = list(value.keys())
    results = []
    all_intervals = list(value.values())
    # Options to pick from
    options = ["specific", "vague"]

    for interval_combination in itertools.product(*all_intervals):

        labels_per_slot = {}
        # Start the loop from 2 for the first slot
        for t in range(1, T + 1):  # Starting from 2 to T inclusive
            # Extract slot t intervals and labels
            print(t)
            slot_intervals = [interval[t - 1] for interval in interval_combination]  # Adjust index to t-1
            # slot_labels = [interval[t - 1] for interval in interval_combination]  # Adjust index to t-1

            variables_dict = {}
            # Loop through each dictionary in the list and store 'variable' in the dictionary
            for i, item in enumerate(slot_intervals):
                # if isinstance(item['variable'], list):  # If 'variable' is a list
                #     variables_dict[labels[i]] = item['variable']
                # else:  # If 'variable' is a single value
                #     variables_dict[labels[i]] = item['variable']
                if isinstance(item['variable'], list):  # If 'variable' is a list
                    # variables_dict[labels[i]] = item['variable']
                    variables_dict[labels[i]] = item
                else:  # If 'variable' is a single value
                    # variables_dict[labels[i]] = item['variable']
                    variables_dict[labels[i]] = item

            overlap = calculate_overlap(slot_intervals)
            if S=='3':  ## this is for not
                print('slot_intervals ---> ',slot_intervals)
                if overlap and slot_intervals[0]['pair_id']!=slot_intervals[1]['pair_id']:
                # if overlap :
                    dic_T={'start': overlap[0],
                            'end': overlap[1] ,
                            'variable':variables_dict ,
                            'type': random.choice(options)}

                    # overlaps.append(overlap)
                    labels_per_slot[f'slot{t}']=dic_T
                else:
                    break  # If any slot has no overlap, skip this combination
            else:
                if overlap:
                    dic_T = {'start': overlap[0],
                             'end': overlap[1],
                             'variable': variables_dict,
                             'type': random.choice(options)}

                    # overlaps.append(overlap)
                    labels_per_slot[f'slot{t}'] = dic_T
                else:
                    break  # If any slot has no overlap, skip this combination

        if len(labels_per_slot) == T:  # Ensure overlaps exist for all T-1 dimensions
            results.append(labels_per_slot)
    return results
def cal_over_laps(value, T, S):
    """
    For each combination of interval‐lists in `value`, compute per‐slot overlaps
    across T slots. If S == '3', only keep overlaps where all pair_ids differ.
    Returns a list of dicts, each with keys 'slot1', …, 'slotT', mapping to:
        { start, end, variable, type }
    """
    def calculate_overlap(intervals):
        if not intervals:
            return None
        try:
            starts = [iv['start'] for iv in intervals]
            ends   = [iv['end']   for iv in intervals]
        except (KeyError, TypeError):
            starts = [iv[0] for iv in intervals]
            ends   = [iv[1] for iv in intervals]
        start, end = max(starts), min(ends)
        return (start, end) if start <= end else None

    labels        = list(value.keys())
    all_intervals = list(value.values())
    options       = ["specific", "vague"]
    results       = []

    # iterate every way to pick one interval‐list per label
    for combo in itertools.product(*all_intervals):
        slot_overlaps = {}

        for t in range(1, T + 1):
            # extract the t-th interval from each label’s list
            slot_iv = [iv[t-1] for iv in combo]
            overlap = calculate_overlap(slot_iv)
            if not overlap:
                break  # no overlap → skip this combo

            # if S == '3', require all pair_ids to be distinct
            # if S == '3':
            #     pair_ids = {iv.get('id_pair') for iv in slot_iv}
            #     if len(pair_ids) < len(slot_iv):
            #         break
            # if S == '3':
            #     pair_ids = [iv.get('id_pair') for iv in slot_iv]
            #     if all(pid == pair_ids[0] for pid in pair_ids):
            #         break  # reject if ALL pair_ids are the same

            # collect the raw interval dicts under each label
            variables = {labels[i]: iv for i, iv in enumerate(slot_iv)}

            slot_overlaps[f"slot{t}"] = {
                "start":    overlap[0],
                "end":      overlap[1],
                "variable": variables,
                "type":     random.choice(options)
            }

        # only keep if we completed all T slots
        if len(slot_overlaps) == T:
            results.append(slot_overlaps)

    return results

def get_slot_order(data):
    slot_order = ''

    for slot, details in data.items():
        # Determine the type based on the value
        type = 'S' if details['type'] == 'specific' else 'V'

        # Append the slot with its corresponding type
        slot_order+=type

    return slot_order
def find_overlap(range1, range2):
    start1, end1 = range1
    start2, end2 = range2

    # Check for overlap
    if start1 <= end2 and start2 <= end1:
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        return (overlap_start, overlap_end)
    else:
        return None

    def compare_matching(ATTs, i, slot_selected_per, flag_slots):

        for key, value1 in ATTs.items():
            if key in ATTs and key in i[slot_selected_per]['variable']:
                if key in ['hhi', 'hhG']:
                    if (
                            ATTs[key]['variable'] == i[slot_selected_per]['variable'][key]['variable'] and
                            ATTs[key]['age_pair'] == i[slot_selected_per]['variable'][key]['age_pair'] and
                            ATTs[key]['race_pair'] == i[slot_selected_per]['variable'][key]['race_pair'] and
                            ATTs[key]['gender_pair'] == i[slot_selected_per]['variable'][key]['gender_pair']
                    ):
                        flag_slots.setdefault(slot_selected_per, {})[key] = True
                    else:
                        flag_slots.setdefault(slot_selected_per, {})[key] = False
                        break
                elif key in ['hoi', 'hoG']:
                    if (
                            ATTs[key]['variable'] == i[slot_selected_per]['variable'][key]['variable'] and
                            ATTs[key]['obj_name'] == i[slot_selected_per]['variable'][key]['obj_name']
                    ):
                        flag_slots.setdefault(slot_selected_per, {})[key] = True
                    else:
                        flag_slots.setdefault(slot_selected_per, {})[key] = False
                else:
                    if ATTs[key]['variable'] == i[slot_selected_per]['variable'][key]['variable']:
                        flag_slots.setdefault(slot_selected_per, {})[key] = True
                    else:
                        flag_slots.setdefault(slot_selected_per, {})[key] = False
                        break
            else:
                flag_slots[slot_selected_per] = False
                break
        return flag_slots
def compare_matching(ATTs, i, slot_selected_per, flag_slots):

        for key, value1 in ATTs.items():
            if key in ATTs and key in i[slot_selected_per]['variable']:
                if key in ['hhi', 'hhG']:
                    if (
                            ATTs[key]['variable'] == i[slot_selected_per]['variable'][key]['variable'] and
                            ATTs[key]['age_pair'] == i[slot_selected_per]['variable'][key]['age_pair'] and
                            ATTs[key]['race_pair'] == i[slot_selected_per]['variable'][key]['race_pair'] and
                            ATTs[key]['gender_pair'] == i[slot_selected_per]['variable'][key]['gender_pair']
                    ):
                        flag_slots.setdefault(slot_selected_per, {})[key] = True
                    else:
                        flag_slots.setdefault(slot_selected_per, {})[key] = False
                        break
                elif key in ['hoi', 'hoG']:
                    if (
                            ATTs[key]['variable'] == i[slot_selected_per]['variable'][key]['variable'] and
                            ATTs[key]['obj_name'] == i[slot_selected_per]['variable'][key]['obj_name']
                    ):
                        flag_slots.setdefault(slot_selected_per, {})[key] = True
                    else:
                        flag_slots.setdefault(slot_selected_per, {})[key] = False
                else:
                    if ATTs[key]['variable'] == i[slot_selected_per]['variable'][key]['variable']:
                        flag_slots.setdefault(slot_selected_per, {})[key] = True
                    else:
                        flag_slots.setdefault(slot_selected_per, {})[key] = False
                        break
            else:
                flag_slots[slot_selected_per] = False
                break
        return flag_slots
def get_match_people_T(all_possible_persons,selected_person,Start_Video, End_Video,T):
    Bucket= {}
    # if selected_person['variation_slots'] is None:
    #    return None
    slot_order = get_slot_order(selected_person)
    # slot_order = 'S'
    if T == 1:
        if slot_order == 'S':
            for person, value in all_possible_persons.items():
                for i in value:
                    flag_slots = {}
                    for slot_selected_per, value_selected_per in selected_person.items():
                        START, END, ATTs = value_selected_per['start'], value_selected_per['end'], value_selected_per[
                            'variable']
                        overlap = find_overlap((i[slot_selected_per]['start'], i[slot_selected_per]['end']),
                                               (START, END))
                        if overlap:  # check overlap
                            flag_slots = compare_matching(ATTs, i, slot_selected_per, flag_slots)  # check feature
                            i[slot_selected_per]['start'] = overlap[0]
                            i[slot_selected_per]['end'] = overlap[1]
                        else:
                            break
                    # Safely check values
                    if flag_slots and all(
                            isinstance(values, dict) and all(values.values()) for values in flag_slots.values()):
                        if person not in Bucket:
                            Bucket[person] = []
                        Bucket[person].append(i)
        else:  # 'V'
            for person, value in all_possible_persons.items():
                for i in value:
                    flag_slots = {}
                    for slot_selected_per, value_selected_per in selected_person.items():
                        START, END, ATTs = Start_Video, End_Video, value_selected_per['variable']
                        flag_slots = compare_matching(ATTs, i, slot_selected_per, flag_slots)  # Only check feature

                    # Safely check values
                    if flag_slots and all(
                            isinstance(values, dict) and all(values.values()) for values in flag_slots.values()):
                        if person not in Bucket:
                            Bucket[person] = []
                        Bucket[person].append(i)

    elif T == 2:

        for person, value in all_possible_persons.items():

            for i in value:

                flag_slots = {}

                valid = True

                for slot_selected_per, value_selected_per in selected_person.items():

                    ATTs = value_selected_per['variable']

                    if slot_order == 'SS':

                        START, END = value_selected_per['start'], value_selected_per['end']

                        overlap = find_overlap((i[slot_selected_per]['start'], i[slot_selected_per]['end']),

                                               (START, END))

                        if overlap:

                            flag_slots = compare_matching(ATTs, i, slot_selected_per, flag_slots)

                            i[slot_selected_per]['start'], i[slot_selected_per]['end'] = overlap

                        else:

                            valid = False

                            break


                    elif slot_order == 'SV':

                        if slot_selected_per == 'slot1':

                            START, END = value_selected_per['start'], value_selected_per['end']

                            overlap = find_overlap((i[slot_selected_per]['start'], i[slot_selected_per]['end']),

                                                   (START, END))

                            if overlap:

                                flag_slots = compare_matching(ATTs, i, slot_selected_per, flag_slots)

                                i[slot_selected_per]['start'], i[slot_selected_per]['end'] = overlap

                            else:

                                valid = False

                                break


                        elif slot_selected_per == 'slot2':

                            START = selected_person['slot1']['end']

                            if START <= i[slot_selected_per]['start']:

                                flag_slots = compare_matching(ATTs, i, slot_selected_per, flag_slots)

                            else:

                                valid = False

                                break


                    elif slot_order == 'VS':

                        if slot_selected_per == 'slot1':

                            END = selected_person['slot2']['start']

                            if i[slot_selected_per]['end'] <= END:

                                flag_slots = compare_matching(ATTs, i, slot_selected_per, flag_slots)

                            else:

                                valid = False

                                break


                        elif slot_selected_per == 'slot2':

                            START, END = value_selected_per['start'], value_selected_per['end']

                            overlap = find_overlap((i[slot_selected_per]['start'], i[slot_selected_per]['end']),

                                                   (START, END))

                            if overlap:

                                flag_slots = compare_matching(ATTs, i, slot_selected_per, flag_slots)

                                i[slot_selected_per]['start'], i[slot_selected_per]['end'] = overlap

                            else:

                                valid = False

                                break


                    elif slot_order == 'VV':

                        flag_slots = compare_matching(ATTs, i, slot_selected_per, flag_slots)

                # Final condition check for all types

                if (

                        valid and

                        flag_slots and

                        all(

                            all(v.values()) if isinstance(v, dict) else bool(v)

                            for v in flag_slots.values()

                        ) and

                        (slot_order != 'VV' or i['slot1']['end'] < i['slot2']['start']) and

                        len(flag_slots) == T

                ):

                    if person not in Bucket:
                        Bucket[person] = []

                    Bucket[person].append(i)

    elif T == 3:
        # For three-slot matching, iterate over each person and candidate
        for person, candidates in all_possible_persons.items():
            for candidate in candidates:
                flag_slots = {}
                # Process each selected slot according to slot_order and slot key
                for slot_key, sel in selected_person.items():
                    # Determine START, END and expected attributes for this slot
                    if slot_order == 'SSS':
                        START, END = sel['start'], sel['end']
                    elif slot_order == 'SVS' and slot_key == 'slot1':
                        START, END = sel['start'], sel['end']
                    elif slot_order == 'SVS' and slot_key == 'slot2':
                        START = selected_person['slot1']['end']
                        END = selected_person['slot3']['start']
                    elif slot_order == 'SVS' and slot_key == 'slot3':
                        START = selected_person['slot1']['end']
                        END = End_Video
                    elif slot_order == 'VSS' and slot_key == 'slot1':
                        START = Start_Video
                        END = selected_person['slot2']['start']
                    elif slot_order == 'VSS' and slot_key in ('slot2', 'slot3'):
                        START, END = sel['start'], sel['end']
                    elif slot_order == 'SSV' and slot_key in ('slot1', 'slot2'):
                        START, END = sel['start'], sel['end']
                    elif slot_order == 'SSV' and slot_key == 'slot3':
                        START = selected_person['slot2']['end']
                        END = End_Video
                    elif slot_order == 'VSV' and slot_key == 'slot1':
                        START = Start_Video
                        END = selected_person['slot2']['start']
                    elif slot_order == 'VSV' and slot_key in ('slot2', 'slot3'):
                        if slot_key == 'slot2':
                            START, END = sel['start'], sel['end']
                        else:
                            START = selected_person['slot2']['end']
                            END = End_Video
                    elif slot_order == 'VVS' and slot_key in ('slot1', 'slot2'):
                        START = Start_Video
                        END = selected_person['slot3']['start']
                    elif slot_order == 'VVS' and slot_key == 'slot3':
                        START, END = sel['start'], sel['end']
                    elif slot_order == 'SVV' and slot_key == 'slot1':
                        START, END = sel['start'], sel['end']
                    elif slot_order == 'SVV' and slot_key in ('slot2', 'slot3'):
                        START = selected_person['slot1']['end']
                        END = End_Video
                    elif slot_order == 'VVV':
                        START, END = Start_Video, End_Video
                    else:
                        # Unknown pattern, skip this candidate
                        flag_slots = {}
                        break

                    ATTs = sel['variable']

                    # Check temporal overlap when needed
                    needs_overlap = slot_order in ('SSS', 'SVS', 'VSS', 'SSV', 'VSV', 'VVS', 'SVV') and (
                                slot_key != 'slot1' or slot_order in ('SSS', 'SVS', 'VSS', 'SSV'))
                    if needs_overlap:
                        overlap = find_overlap((candidate[slot_key]['start'], candidate[slot_key]['end']), (START, END))
                        if not overlap:
                            break
                        # Shrink candidate interval to the overlap for attribute checking
                        candidate[slot_key]['start'], candidate[slot_key]['end'] = overlap

                    # Check attributes for this slot
                    flag_slots = compare_matching(ATTs, candidate, slot_key, flag_slots)

                # Final validation: all slots matched, correct temporal order, and exactly T slots
                if (len(flag_slots) == T and
                        all(all(att_results.values()) for att_results in flag_slots.values()) and
                        candidate['slot1']['end'] < candidate['slot2']['start'] < candidate['slot3']['start']):
                    Bucket.setdefault(person, []).append(candidate)

    return Bucket


def specialize_for_image(person):
    """
    Collapse each [start,end] in person['variation_slots'] (and their nested variables)
    to a single randomly chosen frame, leaving all other fields untouched.
    """
    # Work in-place on the provided dict
    for slot in person.get('variation_slots', {}).values():
        frame = random.randint(slot['start'], slot['end'])
        # sampled_numbers = evenly_sample_items(list(data_h['labels'].keys()), N)
        print(frame)
        slot['start'] = slot['end'] = frame
        for var in slot.get('variable', {}).values():
            var['start'] = var['end'] = frame
    return person

import random

# def pick_random_person(data_all_combinations, attributes_to_use=None):
#     """
#     Randomly selects one person key and one combination of slots (if applicable),
#     then gathers other attributes (e.g., age, race, gender) and returns a dict.
#
#     If attributes_to_use is provided and contains only simple attributes (age, race, gender),
#     then combinations are ignored and an empty dict is used for variation_slots.
#     """
#     # Select a random person key
#     selected_key = random.choice(list(data_all_combinations.keys()))
#
#     # Determine whether to use combination slots
#     simple_attrs = {'age', 'race', 'gender'}
#     use_slots = True
#     if attributes_to_use is not None and set(attributes_to_use).issubset(simple_attrs):
#         use_slots = False
#
#     # Pick a random combination for that key, if needed
#     if use_slots:
#         combinations = data_all_combinations[selected_key].get('combinations', [])
#         if not combinations:
#             raise ValueError(f"No combinations found for key {selected_key}")
#         # Assume each combination is a dict of slot_name -> slot_data
#         selected_combination = random.choice(combinations)
#     else:
#         # No combination slots requested
#         selected_combination = {}
#
#     # Build the result
#     selected_person = {
#         'id': selected_key,
#         'variation_slots': selected_combination
#     }
#
#     # Collect requested attributes
#     for attr, values in data_all_combinations[selected_key].items():
#         if attr == 'combinations':
#             continue
#         if attributes_to_use is not None and attr not in attributes_to_use:
#             continue
#
#         entry = random.choice(values)
#         # Unwrap tuple entries
#         if isinstance(entry, tuple) and entry:
#             entry = entry[0]
#         variable = entry.get('variable')
#         # Unpack single-element lists to scalars
#         if isinstance(variable, list) and len(variable) == 1:
#             variable = variable[0]
#         selected_person[attr] = variable
#
#     return selected_person
#
# import random

import random
import logging

# def pick_random_person(data_all_combinations, attributes_to_use=None):
#     """
#     Randomly selects one person key and one combination of slots,
#     then gathers other attributes (e.g., age, race, gender) and returns a dict.
#
#     Always includes a random variation slot (if available) in 'variation_slots',
#     even when only simple attributes are requested.
#     """
#     print(data_all_combinations)
#     # Validate input dict
#     if not data_all_combinations:
#         raise ValueError("`data_all_combinations` is empty. Ensure it is populated before calling this function.")
#     # Select a random person key
#     keys = list(data_all_combinations.keys())
#     selected_key = random.choice(keys)
#     person_data = data_all_combinations[selected_key]
#
#     # Always pick a slot combination if available
#     combinations = person_data.get('combinations', [])
#     if not combinations:
#         raise ValueError(f"No combinations found for key '{selected_key}'. Available keys: {keys}")
#     selected_combination = random.choice(combinations)
#
#     # Build the result
#     selected_person = {
#         'id': selected_key,
#         'variation_slots': selected_combination
#     }
#
#     # Collect requested simple attributes
#     for attr, values in person_data.items():
#         if attr == 'combinations':
#             continue
#         if attributes_to_use is not None and attr not in attributes_to_use:
#             continue
#
#         # Ensure values list is not empty
#         if not values:
#             logging.warning(f"Attribute list for '{attr}' on key '{selected_key}' is empty; skipping.")
#             continue
#
#         entry = random.choice(values)
#         # Unwrap tuple entries
#         if isinstance(entry, tuple) and entry:
#             entry = entry[0]
#         variable = entry.get('variable') if isinstance(entry, dict) else entry
#         # Unpack single-element lists to scalars
#         if isinstance(variable, list) and len(variable) == 1:
#             variable = variable[0]
#
#         selected_person[attr] = variable
#
#     return selected_person
import random
import logging

def pick_random_person(data_all_combinations, attributes_to_use=None):
    """
    Randomly selects one person key and one combination of slots,
    then gathers other attributes (e.g., age, race, gender) and returns a dict.

    Always includes a random variation slot (if available) in 'variation_slots',
    even when only simple attributes are requested.

    Retries selection if any chosen simple attribute variable equals 'impossible'.
    """
    # Validate input dict
    if not data_all_combinations:
        raise ValueError("`data_all_combinations` is empty. Ensure it is populated before calling this function.")

    keys = list(data_all_combinations.keys())

    while True:
        # Select a random person key
        selected_key = random.choice(keys)
        # selected_key ='h_pedestrian:6'
        person_data = data_all_combinations[selected_key]

        # Always pick a slot combination if available
        combinations = person_data.get('combinations', [])
        if not combinations:
            # Skip this key if no combinations
            logging.warning(f"No combinations found for key '{selected_key}'; retrying selection.")
            continue
        selected_combination = random.choice(combinations)
        # selected_combination = sample_hhi(selected_combination, attributes_to_use)
        # Build the result
        selected_person = {
            'id': selected_key,
            'variation_slots': selected_combination
        }

        # Collect requested simple attributes
        for attr, values in person_data.items():
            if attr == 'combinations':
                continue
            if attributes_to_use is not None and attr not in attributes_to_use:
                continue

            # Ensure values list is not empty
            if not values:
                logging.warning(f"Attribute list for '{attr}' on key '{selected_key}' is empty; skipping.")
                continue

            entry = random.choice(values)
            # Unwrap tuple entries
            if isinstance(entry, tuple) and entry:
                entry = entry[0]
            variable = entry.get('variable') if isinstance(entry, dict) else entry
            # Unpack single-element lists to scalars
            if isinstance(variable, list) and len(variable) == 1:
                variable = variable[0]

            selected_person[attr] = variable

        # Check for 'impossible' in any simple attribute
        invalid = any(
            v == 'impossible'
            for k, v in selected_person.items()
            if k not in ('id', 'variation_slots')
        )
        if invalid:
            # Retry entire selection process
            logging.info(f"Selected attributes contain 'impossible'; retrying selection for key '{selected_key}'.")
            continue

        # Valid selection
        return selected_person



import itertools

def generate_ask_provide_combinations(features, T=None):
    """
    Generate all possible combinations of features to ask and features to provide.

    Args:
        features (list): A list of features.
        T (int): Threshold value. If T > 1, 'age', 'race', and 'gender' should never go to the provide list.

    Returns:
        list: A list of tuples where each tuple contains:
              - ask (list): Features to ask.
              - provide (list): Features to provide.
    """
    protected_features = {'age', 'race', 'gender'}
    results = []

    for r in range(1, len(features)):
        for ask_tuple in itertools.combinations(features, r):
            ask = list(ask_tuple)

            # Use set subtraction to determine provide list
            provide = [f for f in features if f not in ask]

            # If T > 1, ensure 'age', 'race', and 'gender' are not in the provide list
            if T is not None and T > 1 and protected_features.intersection(provide):
                continue

            results.append((ask, provide))

    return results
def get_all_combinations(S, T, target_Q):
    all_combinations = []

    if S == '1':
        if target_Q == 'human':
            base_attributes = ['age', 'race', 'gender', 'action']
            group_pair = ['distance', 'SR_Robot_Ref']
            required_attributes = {'action', 'distance', 'SR_Robot_Ref'}

            # Use a placeholder to represent the grouped attributes
            attributes = base_attributes + ['distance_SR_Robot']

            for r in range(1, len(attributes) + 1):
                combinations = [
                    list(comb) for comb in itertools.combinations(attributes, r)
                    if T == 1 or required_attributes.intersection(set(
                        ['distance' if x == 'distance_SR_Robot' else x for x in comb] +
                        ['SR_Robot_Ref' if x == 'distance_SR_Robot' else x for x in comb]
                    ))
                ]

                # Replace 'distance_SR_Robot' with both attributes
                for comb in combinations:
                    if 'distance_SR_Robot' in comb:
                        comb.remove('distance_SR_Robot')
                        comb.extend(['distance', 'SR_Robot_Ref'])
                all_combinations.extend(combinations)
        elif target_Q == 'object':
            attributes = {
                'obj_name': 'category_R',
                # 'obj_dis_robot': 'D_R',
                # 'SR_robot_ref': 'SR_robot_ref'
            }

            # Create a list of attribute keys
            attribute_keys = list(attributes.keys())

            # Generate combinations for 1 to len(attribute_keys) elements
            all_combinations = []
            for r in range(1, len(attribute_keys) + 1):
                combinations = [list(comb) for comb in itertools.combinations(attribute_keys, r)]
                all_combinations.extend(combinations)

            # Filter combinations to only include those that contain 'category'
            all_combinations = [comb for comb in all_combinations if 'object_category' in comb]

    elif S == '2':  # Pair level
        # if target_Q == 'human':
        #     attributes = ['age', 'race', 'gender', 'hhi','hhg']
        #     required_attributes = {'hhi','hhg'}
        #     for r in range(1, len(attributes) + 1):
        #         combinations = [
        #             list(comb) for comb in itertools.combinations(attributes, r)
        #             if required_attributes.intersection(comb)
        #             and not {'hhi', 'hhg'}.issubset(comb)
        #         ]
        #         all_combinations.extend(combinations)
        if target_Q == 'human':

            attributes = ['age', 'race', 'gender','action', 'hhg', 'hhi']
            all_combinations = []

            for r in range(1, len(attributes) + 1):
                for comb in itertools.combinations(attributes, r):
                    comb_set = set(comb)
                    has_hhg = 'hhg' in comb_set
                    has_hhi = 'hhi' in comb_set

                    # Include if either hhg or hhi is present, but not both
                    if (has_hhg ^ has_hhi):  # XOR: True if only one of them is present
                        all_combinations.append(list(comb))

        elif target_Q == 'human&object':
            attributes = ['age', 'race', 'gender', 'h_dis_robot', 'hoi']
            required_attributes = {'hoi'}
            for r in range(1, len(attributes) + 1):
                combinations = [
                    list(comb) for comb in itertools.combinations(attributes, r)
                    if required_attributes.intersection(comb)
                ]
                all_combinations.extend(combinations)

        elif target_Q == 'object':
            attributes = {
                'category': 'category_R',
                'obj_dis_robot': 'D_R',
                'SR_robot_ref': 'SR_robot_ref',
                'ooG': 'ooG_R'
            }

            # Create a list of attribute keys
            attribute_keys = list(attributes.keys())

            # Generate combinations for 1 to len(attribute_keys) elements
            all_combinations = []
            for r in range(1, len(attribute_keys) + 1):
                combinations = [list(comb) for comb in itertools.combinations(attribute_keys, r)]
                all_combinations.extend(combinations)

            # Filter combinations to include both 'category' and 'ooG'
            all_combinations = [comb for comb in all_combinations if 'category' in comb and 'ooG' in comb]

    elif S == '3':  # Group level
        if target_Q == 'human':
            attributes = ['age', 'race', 'gender', 'hhi', 'hhg']
            required_attributes = {'hhi', 'hhg'}
            for r in range(len(required_attributes), len(attributes) + 1):
                combinations = [
                    list(comb) for comb in itertools.combinations(attributes, r)
                    if required_attributes.issubset(comb)
                ]
                all_combinations.extend(combinations)

        elif target_Q == 'human&object':
            attributes = ['age', 'race', 'gender', 'hhi', 'hhg', 'hoi']
            all_combinations = []

            for r in range(1, len(attributes) + 1):
                for comb in itertools.combinations(attributes, r):
                    comb_set = set(comb)

                    # Include only if 'hoi' is present AND exactly one of 'hhi' or 'hhg' is present
                    if 'hoi' in comb_set and (('hhi' in comb_set) ^ ('hhg' in comb_set)):  # XOR
                        all_combinations.append(list(comb))

    return all_combinations

import ast
import random

import ast
import random

def sample_hhi(selected_combination: dict, attributes_to_use: list):
    for attr in ["hhi", "hhg"]:
        if attr not in attributes_to_use:
            continue

        for slot_key, slot_value in selected_combination.items():
            var_block = slot_value.get("variable", {})
            block = var_block.get(attr)
            if not block:
                continue

            list_str = block["variable"][0]  # Assumes it's a list with a single string
            attr_dict = ast.literal_eval(list_str)

            interaction_list = attr_dict.get("interactions", [])
            if not interaction_list:
                continue

            sampled_interaction = random.choice(interaction_list)
            attr_dict["interactions"] = sampled_interaction

            # Update as a single-item list with stringified dict (to match original format)
            block["variable"] = [str(attr_dict)]

    return selected_combination

def get_final_bbox_length(ti):
    # print(ti)
    try:
        return len(ti['slot1']['final_bbox'])
    except KeyError as e:
        print(f"Missing key in structure: {e}")
        return None



from typing import Any, Dict, List, Tuple, Union

# You’ll need to have these helper functions available in the same module
# (or import them if they live elsewhere):
#   - get_slot_order
#   - find_overlap
#   - compare_matching



#!/usr/bin/env python3
import json
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Union

def fill_data_dict_copy(
    data_dict: Dict[str, Any],
    category: List[Tuple[str, Any]],
    Q_com: str,
    Info: Dict[str, Any],
    target_Q: str,
    Slots: Dict[str, Dict[str, Union[int, str]]]
) -> Dict[str, Any]:
    """
    Populate a deep copy of data_dict with labels, question entities, timestamps, and question text.
    """
    data_dict_copy = deepcopy(data_dict)
    # 1) labels
    data_dict_copy['labels'] = Info

    # 2) question / entities
    question = data_dict_copy.setdefault('question', {})
    entities = question.setdefault('entities', {})
    entities['category'] = target_Q

    def build_hhi_entity(hhi: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'interaction': hhi['variable'],
            'pair': {
                'id': hhi['pair_id'],
                'age': hhi.get('age_pair'),
                'gender': hhi.get('gender_pair'),
                'race': hhi.get('race_pair'),
                'box': hhi['box']
            },
            'type': hhi['type']
        }

    def build_hoi_entity(hoi: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'interaction': hoi['variable'],
            'pair': {
                'track_ID': hoi['pair_id'][0],
                'category_ID': hoi['pair_id'][1],
                'box': hoi['box'],
                'obj_name': hoi['obj_name']
            },
            'type': hoi['type']
        }

    # 3) populate entities
    for key, val in category:
        if val is None:
            continue
        if key == 'hhi':
            entities[key] = build_hhi_entity(val)
        elif key == 'hoi':
            entities[key] = build_hoi_entity(val)
        elif key == 'changed_attribute':
            ch_attr: Dict[str, Any] = {}
            for slot_name, slot_info in val.items():
                slot_entry: Dict[str, Any] = {}
                for attr_key, attr_val in slot_info.items():
                    if attr_key == 'variable':
                        var_entry: Dict[str, Any] = {}
                        for inner_key, inner_val in attr_val.items():
                            if inner_key in ('hhi', 'hoi'):
                                builder = build_hhi_entity if inner_key == 'hhi' else build_hoi_entity
                                var_entry[inner_key] = builder(inner_val)
                            else:
                                var_entry[inner_key] = inner_val
                        slot_entry['variable'] = var_entry
                    else:
                        slot_entry[attr_key] = attr_val
                ch_attr[slot_name] = slot_entry
            entities[key] = ch_attr
        else:
            entities[key] = val

    # 4) timestamps
    timestamps = question.setdefault('timestamps', {})
    for slot_name, slot_info in Slots.items():
        start, end, typ = slot_info['start'], slot_info['end'], slot_info['type']
        timestamps[slot_name] = {
            't1': float(start),
            't2': float(end),
            'image_ids': [f"{i:06}.jpg" for i in range(start, end + 1)],
            'type': typ
        }

    # 5) question text
    question['question'] = Q_com

    return data_dict_copy


def format_descriptors(attribute_strings):
    if not attribute_strings:
        return ""

    # Flatten list if accidentally passed as list of lists
    flat_list = []
    for attr in attribute_strings:
        if isinstance(attr, list):
            flat_list.extend(attr)
        else:
            flat_list.append(attr)

    attribute_strings = [attr for attr in flat_list if attr is not None]

    if len(attribute_strings) == 0:
        return ""
    elif len(attribute_strings) == 1:
        return attribute_strings[0]
    elif len(attribute_strings) == 2:
        # Filter out empty strings
        non_empty = [s for s in attribute_strings if len(s) > 0]
        # Decide output based on number of non-empty strings
        if len(non_empty) == 2:
            result = " and ".join(non_empty)
        elif len(non_empty) == 1:
            result = non_empty[0]
        else:
            result = ""
        return  result
        # return " and ".join(attribute_strings)
    else:
        return ", ".join(attribute_strings[:-1]) + " and " + attribute_strings[-1]


def format_descriptors_then(attribute_strings):
    if not attribute_strings:
        return ""
    # print(attribute_strings)
    # Filter out None values
    attribute_strings = [attr for attr in attribute_strings if attr is not None]

    if len(attribute_strings) == 0:
        return ""  # Return an empty string if no valid attributes remain
    elif len(attribute_strings) == 1:
        return attribute_strings[0]
    elif len(attribute_strings) == 2:
        return " and then ".join(attribute_strings)
    else:
        return ", then ".join(attribute_strings[:-1]) + " and then " + attribute_strings[-1]
def number_to_words(n):
    num_words = {
        0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
        6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten',
        11: 'eleven', 12: 'twelve', 13: 'thirteen', 14: 'fourteen', 15: 'fifteen',
        16: 'sixteen', 17: 'seventeen', 18: 'eighteen', 19: 'nineteen',
        20: 'twenty', 30: 'thirty', 40: 'forty', 50: 'fifty',
        60: 'sixty', 70: 'seventy', 80: 'eighty', 90: 'ninety'
    }

    if n in num_words:
        return num_words[n]
    elif 21 <= n <= 99:
        tens = (n // 10) * 10
        ones = n % 10
        return num_words[tens] + ('-' + num_words[ones] if ones else '')
    else:
        return "Number out of range"


# def Refinement_attribute_person_T(attributes, bbox_number):
#     """
#     Build a descriptive sentence for one or more pedestrians given their attributes.
#
#     - attributes: either a dict or a list of (label, value) tuples.
#     - bbox_number: number of bounding boxes (1 → singular “person”, >1 → plural “persons”).
#     """
#     # 1) If they passed in a list of (label, value) pairs, turn it into a dict
#     if isinstance(attributes, list):
#         attributes = dict(attributes)
#
#     # 2) Filter out any attributes with falsy values (None, empty, etc.)
#     attributes = {lbl: val for lbl, val in attributes.items() if val}
#
#     if not attributes:
#         return "No relevant attributes found."
#
#     # --- initialize containers ---------------------------------------------
#     age = set()
#     race = set()
#     gender = set()
#     group_size = set()
#
#     h_dis_robot = set()
#     sr_robot_ref = set()
#     hhi_set = set()
#     hhg_set = set()
#     hoi_set = set()
#     hog_set = set()
#
#     # --- basic attributes (gender, race, age, group_size) -------------------
#     for lbl, val in attributes.items():
#         if lbl == "gender" and val != "impossible":
#             gender.add(val)
#         elif lbl == "race" and val != "impossible":
#             race.add(val)
#         elif lbl == "age" and val != "impossible":
#             age.add(val)
#         elif lbl == "group_size" and val != "impossible":
#             group_size.add(val)
#
#     # format descriptors like “young, Asian, female”
#     combined_list1 = list(age) + list(race) + list(gender)
#     part1 = format_descriptors(combined_list1)
#
#     # handle group size (“alone” or “part of a group of three”)
#     group_size_list = []
#     if group_size:
#         gs = next(iter(group_size))
#         if gs == 1:
#             group_size_list = ["alone"]
#         else:
#             group_size_list = [f"part of a group of {number_to_words(gs)}"]
#
#     # --- check for any ‘changed_attribute’ flag ----------------------------
#     flg_only_GAE = True
#     slots = attributes.get('variation_slots', {})
#     for slot_name, slot_info in slots.items():
#         var = slot_info.get('variable')
#         if var:  # non-empty dicts evaluate to True
#             flg_only_GAE = False
#
#     # --- process each variation slot (actions, interactions, robot refs) ----
#     list_slots = []
#     for slot_name, slot_info in attributes.get("variation_slots", {}).items():
#         start, end = slot_info["start"], slot_info["end"]
#         slot_type = slot_info.get("type", "vague")
#
#         actions = set()
#         # (re-)clear these per-slot
#         h_dis_robot.clear()
#         sr_robot_ref.clear()
#         hhi_set.clear()
#         hoi_set.clear()
#
#         for K, V in slot_info.get("variable", {}).items():
#             if K == "action":
#                 actions.update(V.get("variable", []))
#
#             if K == "distance":
#                 h_dis_robot.add(V.get("variable", "")[0] + " meters")
#             if K == "SR_Robot_Ref":
#                 sr_robot_ref.add(V.get("variable", "")[0])
#
#             if 'hhi' == K:
#                 # index = attributes_picked.index('hhi')
#                 # hhi = list(Attributes_changed[index])[0]
#                 # hhi = V[0]
#                 HHI = V.get('variable', '')  # Safely get the 'variable' key
#                 # hhi=set(hhi)
#                 HHI2=[HHI.get('interactions', '')]
#                 list_HHi = []
#                 for hhi in HHI2:
#                     if "together" in hhi:
#                         value1 = hhi.replace("together", "")
#                     elif hhi == "walking toward each other":
#                         value1 = "walking toward"
#                     elif hhi == "conversation":
#                         value1 = "chatting"
#                     elif hhi == "looking at sth":
#                         value1 = "looking at something"
#                     # else:
#                     #     value1 = f"{hhi} with someone"
#                     list_HHi.append(value1)
#                 formatted_pair_interaction = format_descriptors(list_HHi)
#
#                 # print(V)  # Replace value with V or initialize value properly
#                 if V.get('type') == 'specific':  # Safely check 'type'
#                     pair_hhi = [HHI['age_pair'], HHI['gender_pair'], HHI['race_pair']]
#                     formatted_pair = format_descriptors(pair_hhi)  # Ensure this function works as intended
#                     if 'toward' in formatted_pair_interaction:
#                         value1 = f"{formatted_pair_interaction} a {formatted_pair} person"
#                     else:
#                         value1 = f"{formatted_pair_interaction} with a {formatted_pair} person"
#                 else:
#                     if 'toward' in formatted_pair_interaction:
#                         value1 = f"{formatted_pair_interaction} a someone"
#                     else:
#                         value1 = f"{formatted_pair_interaction} with someone"
#
#                 hhi_set.add(value1)
#
#             if 'hhG' == K:
#                 # index = attributes_picked.index('hhi')
#                 # hhi = list(Attributes_changed[index])[0]
#                 # hhi = V[0]
#                 HHG = V.get('variable', '')  # Safely get the 'variable' key
#                 distance = HHG.get('Distance', '')
#                 list_values = [HHG.get('SR_Person_Ref', ''), f'{distance} distance']
#
#                 list_pair_attr = [value['variable']['hhG']['gender_pair'], value['variable']['hhG']['age_pair'],
#                                   value['variable']['hhG']['race_pair']]
#
#                 hhg_set.add(
#                     f"a {format_descriptors(list_pair_attr)} person is located to his/her/their {format_descriptors(list_values)}")
#                 # hhg_set.add(f"a {format_descriptors(list_pair_attr)} person is located to his/her/their {HHG.get('SR_Person_Ref', '')}")
#
#             if 'hoi' == K:
#                 HOI = V.get('variable', '')  # Safely get the 'variable' key
#                 # hhi=set(hhi)
#                 list_HOi = []
#                 for hoi in HOI:
#
#                     if hoi == 'working':
#                         hoi = 'working with'  # Modify the first element
#
#                     if V['type'] == 'specific':
#                         if hoi in all_human_pose_action:
#                             hoi = hoi + ' on ' + V['obj_name']
#                         else:
#                             hoi = hoi + ' ' + V['obj_name']
#                     else:
#                         if hoi in all_human_pose_action:
#                             hoi = hoi + ' on something'
#                         else:
#                             hoi = hoi + ' something'
#                     if hoi not in list_HOi:
#                         list_HOi.append(hoi)
#
#                 formatted_pair_interaction = format_descriptors(list_HOi)
#                 hoi_set.add(formatted_pair_interaction)
#
#             if 'hoG' == K:
#                 # index = attributes_picked.index('hhi')
#                 # hhi = list(Attributes_changed[index])[0]
#                 # hhi = V[0]
#                 HoG = V.get('variable', '')  # Safely get the 'variable' key
#                 distance = HoG.get('Distance', '')
#                 list_values = [HoG.get('SR_Person_Ref', ''), f'{distance} distance']
#
#                 # list_pair_attr = [value['variable']['hhG']['gender_pair'], value['variable']['hhG']['age_pair'], value['variable']['hhG']['race_pair']]
#
#                 hog_set.add(
#                     f"a {value['variable']['hoG']['obj_name']} is located to his/her/their {format_descriptors(list_values)}")
#                 # hhg_set.add(f"a {format_descriptors(list_pair_attr)} person is located to his/her/their {HHG.get('SR_Person_Ref', '')}")
#
#         # build the time window
#         t1 = f"{start / 15:.2f}".rstrip("0").rstrip(".")
#         t2 = f"{end / 15:.2f}".rstrip("0").rstrip(".")
#
#         # robot location phrasing
#         locateds = []
#         combo = list(h_dis_robot) + list(sr_robot_ref)
#         if combo:
#             locateds = [f"located at {format_descriptors(combo)} relative to me(robot)"]
#
#         # combine everything for this slot
#         combined_actions = list(actions) + group_size_list + locateds
#         desc = format_descriptors(combined_actions)
#
#         # --- here is the updated logic for "at" or "during" -------------------
#         if slot_type == "specific":
#             if t1 == t2:
#                 # list_slots.append(f"{desc} at {t1} seconds")
#                 list_slots.append(f"{desc}")
#             else:
#                 # list_slots.append(f"{desc} during {t1} to {t2} seconds")
#                 list_slots.append(f"{desc}")
#         else:
#             list_slots.append(desc)
#
#     part2 = format_descriptors_then(list_slots)
#
#     # --- assemble final sentence -------------------------------------------
#     if not flg_only_GAE:
#         if part2:
#             if bbox_number > 1:
#                 sentence = f"Find {part1} persons who are {part2}"
#             else:
#                 sentence = f"Find {part1} person who is {part2}"
#         else:
#             if bbox_number > 1:
#                 sentence = f"Find {part1} persons"
#             else:
#                 sentence = f"Find {part1} person"
#     else:
#         # changed‐attribute special case
#         if bbox_number > 1:
#             sentence = f"Find {part1} persons {part2}"
#         else:
#             sentence = f"Find {part1} person {part2}"
#
#     return sentence.replace("  ", " ")
def Refinement_attribute_person_T(attributes, bbox_number):
    """
    Build a descriptive sentence for one or more pedestrians given their attributes.

    - attributes: either a dict or a list of (label, value) pairs.
    - bbox_number: number of bounding boxes (1 → singular “person”, >1 → plural).
    """
    # Normalize input to a dict
    items = attributes if isinstance(attributes, dict) else dict(attributes)

    # Basic descriptors
    age = items.get('age')
    race = items.get('race')
    gender = items.get('gender')
    group_size = items.get('group_size')

    age_set = {age} if age and age != 'impossible' else set()
    race_set = {race} if race and race != 'impossible' else set()
    gender_set = {gender} if gender and gender != 'impossible' else set()

    # Compose part1
    part1 = format_descriptors(list(age_set) + list(race_set) + list(gender_set))
    if group_size and group_size != 'impossible':
        part1 += ' alone' if group_size == 1 else f" part of a group of {number_to_words(group_size)}"

    # Process variation_slots
    var_slots = items.get('variation_slots', {}) or {}
    fragments = []
    for slot_info in var_slots.values():
        if not isinstance(slot_info, dict):
            continue
        start = slot_info.get('start')
        end = slot_info.get('end')
        slot_type = slot_info.get('type', 'vague')
        var_map = slot_info.get('variable', {}) or {}

        parts = []
        # Generic actions
        if 'action' in var_map:
            parts += var_map['action'].get('variable', [])

        # Robot-relative
        if 'distance' in var_map:
            d = var_map['distance'].get('variable', '')[0]
            parts.append(f"located to {d} distance")
        if 'SR_Robot_Ref' in var_map:
            SR_robot = var_map['SR_Robot_Ref'].get('variable', '')[0]
            parts.append(f"{SR_robot} relative to me(robot)")

        # Human-human interaction (hhi)
        if 'hhi' in var_map:
            raw_list = var_map['hhi'].get('variable') or []
            raw_list = [raw_list] if isinstance(raw_list, str) else raw_list
            norm = []
            for h in raw_list:
                text = h.replace('together', '').strip() if 'together' in h else h
                if h == 'walking toward each other':
                    text = 'walking toward'
                elif h == 'conversation':
                    text = 'chatting'
                elif h == 'looking at sth':
                    text = 'looking at something'
                norm.append(text)
            desc = format_descriptors(norm)
            pair_attrs = [var_map['hhi'].get('age_pair'), var_map['hhi'].get('gender_pair'), var_map['hhi'].get('race_pair')]
            desc += f" with a {format_descriptors(pair_attrs)} person"
            parts.append(desc)

        # Human-human geometry (hhg)
        if 'hhg' in var_map:
            raw_list = var_map['hhg'].get('variable') or []
            raw_list = [raw_list] if isinstance(raw_list, str) else raw_list
            desc = format_descriptors(raw_list)
            pair_attrs = [var_map['hhg'].get('age_pair'), var_map['hhg'].get('gender_pair'), var_map['hhg'].get('race_pair')]
            desc1 = f"a {format_descriptors(pair_attrs)} person located to his\her {desc} distance"
            parts.append(desc1)

        # Human-object interaction (hoi)
        if 'hoi' in var_map:
            raw_list = var_map['hoi'].get('variable') or []
            raw_list = [raw_list] if isinstance(raw_list, str) else raw_list
            desc = format_descriptors(raw_list)
            obj = var_map['hoi'].get('obj_name')
            if obj:
                obj = to_singular(obj)
                desc += f" a {obj}"
            parts.append(desc)

        # Build fragment
        fragment = format_descriptors(parts)
        if slot_type == 'specific' and isinstance(start, (int, float)) and isinstance(end, (int, float)):
            t1 = f"{start/15:.2f}".rstrip('0').rstrip('.')
            t2 = f"{end/15:.2f}".rstrip('0').rstrip('.')
            fragment = f"{fragment} during {t1} to {t2} seconds" if t1 != t2 else fragment
        fragments.append(fragment)

    # Compose part2
    part2 = format_descriptors_then(fragments)

    # Final assembly
    if part2:
        if bbox_number > 1:
            sentence = f"Find {part1} persons who are {part2}"
        else:
            sentence = f"Find {part1} person who is {part2}"
    else:
        sentence = f"Find {part1} {'persons' if bbox_number>1 else 'person'}"

    return sentence.replace('  ', ' ')

def Refinement_attribute_person_T_VQA_Wh(attributes_dict, ask, provides, T, len_answer):
    # --- Helper: Convert dict format to expected attribute list ---
    attributes = []

    for attr in ['age', 'race', 'gender', 'group_size']:
        value = attributes_dict.get(attr)
        if value:  # Skip if None or empty
            attributes.append((attr, value))

    if 'variation_slots' in attributes_dict:
        attributes.append(('changed_attribute', attributes_dict['variation_slots']))

    # --- Step 1: Map ask fields ---
    refine_ask = [VQA_corresponding[A] for A in ask]

    if not attributes:
        return "No relevant attributes found.",

    selected_attributes = attributes

    # --- Step 2: Extract person descriptors (age, race, gender, group size) ---
    attribute_strings = []
    age, race, gender, group_size = set(), set(), set(), set()
    num = 0

    for lbl, value in selected_attributes:
        if lbl == "gender" and lbl in provides and value != 'impossible':
            gender.add(value)
            num += 1
        elif lbl == "race" and lbl in provides and value != 'impossible':
            race.add(value)
            num += 1
        elif lbl == "age" and lbl in provides and value != 'impossible':
            age.add(value)
            num += 1
        elif lbl == "group_size" and lbl in provides and value != 'impossible':
            group_size.add(value)
            num += 1

    combined_list1 = list(age | race | gender)
    part1 = format_descriptors(combined_list1)

    group_size_list = list(group_size)
    if group_size_list:
        if group_size_list[0] == 1:
            group_size_list = ['alone']
        else:
            group_size_list = [f'part of a group of {number_to_words(group_size_list[0])}']

    # --- Step 3: Process changed_attribute (variation_slots) ---
    list_slots = []

    for slot, value in dict(selected_attributes).get('changed_attribute', {}).items():
        start = value['start']
        end = value['end']
        variable = value['variable']
        type = value['type']

        actions, h_dis_robot, sr_robot_ref = set(), set(), set()
        hhi_set, hhg_set, hoi_set, hog_set = set(), set(), set(), set()

        for K, V in variable.items():
            if K == 'action' and K in provides:
                actions.add(V.get('variable', [''])[0])
            elif K == 'distance' and K in provides:
                h_dis_robot.add(V.get('variable', '')[0] + ' distance')
            elif K == 'SR_Robot_Ref' and K in provides:
                sr_robot_ref.add(V.get('variable', '')[0])

            elif K == 'hhi' and K in provides:
                HHI = V.get('variable', [])
                list_HHi = []
                for hhi in HHI:
                    if "together" in hhi:
                        value1 = hhi.replace("together", "")
                    elif hhi == "walking toward each other":
                        value1 = "walking toward"
                    elif hhi == "conversation":
                        value1 = "chatting"
                    elif hhi == "looking at sth":
                        value1 = "looking at something"
                    else:
                        value1 = hhi
                    list_HHi.append(value1)

                formatted_pair_interaction = format_descriptors(list_HHi)
                if V.get('type') == 'specific':
                    pair_hhi = [V['age_pair'], V['gender_pair'], V['race_pair']]
                    formatted_pair = format_descriptors(pair_hhi)
                    if 'toward' in formatted_pair_interaction:
                        value1 = f"{formatted_pair_interaction} a {formatted_pair} person"
                    else:
                        value1 = f"{formatted_pair_interaction} with a {formatted_pair} person"
                else:
                    if 'toward' in formatted_pair_interaction:
                        value1 = f"{formatted_pair_interaction} someone"
                    else:
                        value1 = f"{formatted_pair_interaction} with someone"
                hhi_set.add(value1)

            elif K == 'hhg' and K in provides:
                HHG = V.get('variable', {})
                distance = HHG.get('Distance', '')
                list_values = [HHG.get('SR_Person_Ref', ''), f'{distance} distance']
                list_pair_attr = [V['gender_pair'], V['age_pair'], V['race_pair']]
                hhg_set.add(f"a {format_descriptors(list_pair_attr)} person is located to his/her/their {format_descriptors(list_values)}")

            elif K == 'hoi' and K in provides:
                HOI = V.get('variable', [])
                list_HOi = []
                for hoi in HOI:
                    if hoi == 'working':
                        hoi = 'working with'
                    if V['type'] == 'specific':
                        if hoi in all_human_pose_action:
                            hoi = hoi + ' on ' + V['obj_name']
                        else:
                            hoi = hoi + ' ' + V['obj_name']
                    else:
                        if hoi in all_human_pose_action:
                            hoi = hoi + ' on something'
                        else:
                            hoi = hoi + ' something'
                    if hoi not in list_HOi:
                        list_HOi.append(hoi)

                formatted_pair_interaction = format_descriptors(list_HOi)
                hoi_set.add(formatted_pair_interaction)

            elif K == 'hog' and K in provides:
                HoG = V.get('variable', {})
                distance = HoG.get('Distance', '')
                list_values = [HoG.get('SR_Person_Ref', ''), f'{distance} distance']
                hog_set.add(f"a {V['obj_name']} is located to his/her/their {format_descriptors(list_values)}")

        # Combine all sub-attributes
        locateds = []
        combined_list = list(h_dis_robot | sr_robot_ref)
        if combined_list:
            locateds = [f"located at {format_descriptors(combined_list)} relative to me(robot)"]

        hhi_hoi_hhg_hog = format_descriptors([list(hhi_set | hoi_set | hhg_set | hog_set)])
        combined_list2 = list(actions) + group_size_list + [hhi_hoi_hhg_hog] + locateds

        t1 = f"{start / 15:.2f}".rstrip('0').rstrip('.')
        t2 = f"{end / 15:.2f}".rstrip('0').rstrip('.')

        if type == 'specific':
            if T > 1:
                part_slot = f"{format_descriptors(combined_list2)} during {t1} to {t2} seconds"
            else:
                part_slot = f"{format_descriptors(combined_list2)}"
        elif type == 'vague':
            part_slot = f"{format_descriptors(combined_list2)}"
        list_slots.append(part_slot)

    part2 = format_descriptors_then(list_slots)

    # --- Step 4: Build the final question ---
    allowed_items = {'age', 'race', 'gender'}
    other_items = [item for item in refine_ask if item not in allowed_items]

    if T > 1 and other_items:
        if len_answer == 1:
            question = f"What is the {format_descriptors(refine_ask)} order of the persons who are {part2}?"
        else:
            question = f"What are the {format_descriptors(pluralize_words(refine_ask))} order of the persons who are {part2}?"
    else:
        if len_answer == 1:
            question = f"What is the {format_descriptors(refine_ask)} of the persons who are {part2}?"
        else:
            question = f"What are the {format_descriptors(pluralize_words(refine_ask))} of the persons who are {part2}?"

    return question

def pluralize_words(word_list):
    """
    Convert singular words in the list to their plural forms, handling special cases.
    :param word_list: List of singular words.
    :return: List of plural words.
    """
    special_cases = {
        'distance relative to me(robot)': 'distances relative to me(robot)',
        'spatial position relative to me(robot)': 'spatial positions relative to me(robot)',
        'social interaction': 'social interactions',
        'geometry between people': 'geometry between people',
        'interaction between person and object': 'interactions between person and object',
        # 'hoG': 'geometry between person and object'
    }

    pluralizer = Pluralizer()
    return [special_cases.get(word, pluralizer.plural(word)) for word in word_list]
def to_singular(word):
    p = inflect.engine()
    if p.singular_noun(word):  # Returns False if not plural, otherwise returns the singular form
        return p.singular_noun(word)
    return word


def build_single_attribute_descriptions(attributes):
    """
    For each individual attribute in the input, generate a description sentence.
    Returns a dictionary like: {'race: Asian': 'Find Asian person', ...}
    """
    result = {}

    base_info = {
        k: v for k, v in attributes.items()
        if k not in ['variation_slots', 'race', 'age', 'gender', 'group_size']
    }

    # Top-level singular attributes
    for key in ['race', 'age', 'gender', 'group_size']:
        if key in attributes:
            new_attr = dict(base_info)
            new_attr[key] = attributes[key]
            new_attr['variation_slots'] = {}
            # key_label = f"{key}: {attributes[key]}"
            key_label = f"{key}"
            sentence = Refinement_attribute_person_T(new_attr, 1)
            result[key_label] = sentence

    # From variation_slots
    var_slots = attributes.get('variation_slots', {})
    for slot_name, slot in var_slots.items():
        for var_key, var_value in slot.get('variable', {}).items():
            new_attr = dict(base_info)
            new_attr['variation_slots'] = {
                slot_name: {
                    'start': slot.get('start'),
                    'end': slot.get('end'),
                    'type': slot.get('type', 'vague'),
                    'variable': {
                        var_key: var_value
                    }
                }
            }

            # Build label from first value
            val = var_value.get('variable')
            if isinstance(val, list):
                val_text = val[0]
            else:
                val_text = str(val)
            # key_label = f"{var_key}: {val_text}"
            key_label = f"{var_key}"

            sentence = Refinement_attribute_person_T(new_attr, 1)
            result[key_label] = sentence

    return result


# def build_ordered_attribute_descriptions(attributes, ordered_keys):
#     """
#     Build descriptions by incrementally adding attributes in a fixed order.
#     Example output:
#         {
#             'age': 'Find young person',
#             'age, action': 'Find young person who is walking',
#             'age, action, gender': 'Find young and female person who is walking'
#         }
#     """
#     result = {}
#
#     # Define the order in which attributes should be added
#     # ordered_keys = ['age', 'action', 'gender']
#     # ordered_keys = ['age', 'action', 'gender']
#
#     # Extract top-level attributes
#     top_attrs = {k: attributes[k] for k in ['age', 'gender', 'race', 'group_size'] if k in attributes}
#
#     # Extract from variation_slots (e.g., action)
#     var_attrs = {}
#     var_slots = attributes.get('variation_slots', {})
#     for slot in var_slots.values():
#         for var_key, var_value in slot.get('variable', {}).items():
#             val = var_value.get('variable')
#             val_text = val[0] if isinstance(val, list) else str(val)
#             var_attrs[var_key] = val_text
#
#     all_attrs = {**top_attrs, **var_attrs}
#     base_info = {k: v for k, v in attributes.items() if
#                  k not in ['variation_slots', 'race', 'age', 'gender', 'group_size']}
#
#     current_combo = {}
#     for i in range(len(ordered_keys)):
#         key_subset = ordered_keys[:i + 1]
#         combo_key = ', '.join(key_subset)
#
#         # Build combined attribute dict
#         combo_attr = dict(base_info)
#         if 'action' in key_subset and 'action' in var_attrs:
#             combo_attr['variation_slots'] = {
#                 'slot1': {
#                     'start': 0,
#                     'end': 0,
#                     'type': 'specific',
#                     'variable': {
#                         'action': {'variable': [var_attrs['action']]}
#                     }
#                 }
#             }
#         for k in key_subset:
#             if k in top_attrs:
#                 combo_attr[k] = top_attrs[k]
#
#         # Generate sentence
#         sentence = Refinement_attribute_person_T(combo_attr, 1)
#         result[combo_key] = sentence
#
#     return result
# def build_ordered_attribute_descriptions(attributes, ordered_keys):
#     """
#     Build descriptions by incrementally adding attributes in a fixed order.
#     """
#     result = {}
#
#     # Separate top-level and variation-slot attributes
#     top_attrs = {k: attributes[k] for k in ['age', 'gender', 'race', 'group_size'] if k in attributes}
#
#     # Get variation attributes like 'hhi', 'action', etc.
#     var_attrs = {}
#     var_slots = attributes.get('variation_slots', {})
#     for slot in var_slots.values():
#         for var_key, var_value in slot.get('variable', {}).items():
#             val = var_value.get('variable')
#             val_text = val[0] if isinstance(val, list) else str(val)
#             var_attrs[var_key] = val_text
#
#     base_info = {k: v for k, v in attributes.items() if k not in ['variation_slots', 'race', 'age', 'gender', 'group_size']}
#
#     for i in range(len(ordered_keys)):
#         key_subset = ordered_keys[:i + 1]
#         combo_key = ', '.join(key_subset)
#
#         # Create new attribute dict
#         combo_attr = dict(base_info)
#
#         # Add top-level attributes
#         for k in key_subset:
#             if k in top_attrs:
#                 combo_attr[k] = top_attrs[k]
#
#         # Add variation attributes (e.g. action, hhi, hhg, etc.)
#         variation_variables = {}
#         for k in key_subset:
#             if k in var_attrs:
#                 variation_variables[k] = {'variable': [var_attrs[k]]}
#
#         if variation_variables:
#             combo_attr['variation_slots'] = {
#                 'slot1': {
#                     'start': 0,
#                     'end': 0,
#                     'type': 'specific',
#                     'variable': variation_variables
#                 }
#             }
#
#         # Generate sentence
#         sentence = Refinement_attribute_person_T(combo_attr, 1)
#         result[combo_key] = sentence
#
#     return result
# def build_ordered_attribute_descriptions(attributes, ordered_keys):
#     """
#     Build descriptions for each single attribute in ordered_keys independently.
#     """
#     result = {}
#
#     # Separate top-level attributes (age, gender, race, group_size)
#     top_attrs = {k: attributes[k] for k in ['age', 'gender', 'race', 'group_size'] if k in attributes}
#
#     # Extract variation variables like 'hhi', 'hhg', etc.
#     var_attrs = {}
#     var_slots = attributes.get('variation_slots', {})
#     for slot in var_slots.values():
#         for var_key, var_value in slot.get('variable', {}).items():
#             val = var_value.get('variable')
#             val_text = val[0] if isinstance(val, list) else str(val)
#             var_attrs[var_key] = val_text
#
#     # Keep other base info except the handled keys
#     base_info = {k: v for k, v in attributes.items() if k not in ['variation_slots', 'race', 'age', 'gender', 'group_size']}
#
#     for i in range(len(ordered_keys)):
#         # Pick one key at a time as a list for compatibility
#         key_subset = [ordered_keys[i]]
#         combo_key = ordered_keys[i]
#
#         # Create a fresh attribute dictionary
#         combo_attr = dict(base_info)
#
#         # Add top-level attributes if in key_subset
#         for k in key_subset:
#             if k in top_attrs:
#                 combo_attr[k] = top_attrs[k]
#
#         # Add variation attributes if in key_subset
#         variation_variables = {}
#         for k in key_subset:
#             if k in var_attrs:
#                 variation_variables[k] = {'variable': [var_attrs[k]]}
#
#         # Add variation_slots if any variation variables exist
#         if variation_variables:
#             # Copy start and end from original variation_slots (using first slot)
#             original_slot = next(iter(var_slots.values()), None)
#             start = original_slot.get('start', 0) if original_slot else 0
#             end = original_slot.get('end', 0) if original_slot else 0
#
#             combo_attr['variation_slots'] = {
#                 'slot1': {
#                     'start': start,
#                     'end': end,
#                     'type': 'specific',
#                     'variable': variation_variables
#                 }
#             }
#
#         # Generate sentence with your external function
#         sentence = Refinement_attribute_person_T(combo_attr, 1)
#         result[combo_key] = sentence
#
#     return result

def build_ordered_attribute_descriptions(attributes, ordered_keys, incremental=True):
    """
    Build descriptions either incrementally adding attributes or one at a time.

    Parameters:
    - attributes: dict with all attributes including variation_slots
    - ordered_keys: list of keys to include
    - incremental: if True, build cumulative subsets; if False, build single keys
    """
    result = {}

    # Separate top-level attributes
    top_attrs = {k: attributes[k] for k in ['age', 'gender', 'race', 'group_size'] if k in attributes}

    # Extract variation variables like 'hhi', 'hhg', etc.
    var_attrs = {}
    var_slots = attributes.get('variation_slots', {})
    for slot in var_slots.values():
        for var_key, var_value in slot.get('variable', {}).items():
            val = var_value.get('variable')
            val_text = val[0] if isinstance(val, list) else str(val)
            var_attrs[var_key] = val_text

    base_info = {k: v for k, v in attributes.items() if k not in ['variation_slots', 'race', 'age', 'gender', 'group_size']}

    for i, key in enumerate(sorted(ordered_keys)):
        if incremental:
            # Incremental mode: cumulative subsets
            key_subset = ordered_keys[:i + 1]
            combo_key = ', '.join(key_subset)
        else:
            # Single mode: one key at a time as list
            key_subset = [ordered_keys[i]]
            combo_key = ordered_keys[i]

        combo_attr = dict(base_info)

        for k in key_subset:
            if k in top_attrs:
                combo_attr[k] = top_attrs[k]

        variation_variables = {}
        for k in key_subset:
            if k in var_attrs:
                variation_variables[k] = {'variable': [var_attrs[k]]}

        if variation_variables:
            original_slot = next(iter(var_slots.values()), None)
            start = original_slot.get('start', 0) if original_slot else 0
            end = original_slot.get('end', 0) if original_slot else 0

            combo_attr['variation_slots'] = {
                'slot1': {
                    'start': start,
                    'end': end,
                    'type': 'specific',
                    'variable': variation_variables
                }
            }

        sentence = Refinement_attribute_person_T(combo_attr, 1)
        result[combo_key] = sentence

    return result

def build_ordered_attribute_descriptions_VQA(attributes, ask, provide, T, incremental=True):
    """
    Build descriptions either incrementally adding attributes or one at a time.

    Parameters:
    - attributes: dict with all attributes including variation_slots
    - ordered_keys: list of keys to include
    - incremental: if True, build cumulative subsets; if False, build single keys
    """
    ordered_keys = ask

    result = {}

    # Separate top-level attributes
    top_attrs = {k: attributes[k] for k in ['age', 'gender', 'race', 'group_size'] if k in attributes}

    # Extract variation variables like 'hhi', 'hhg', etc.
    var_attrs = {}
    var_slots = attributes.get('variation_slots', {})
    for slot in var_slots.values():
        for var_key, var_value in slot.get('variable', {}).items():
            val = var_value.get('variable')
            val_text = val[0] if isinstance(val, list) else str(val)
            var_attrs[var_key] = val_text

    base_info = {k: v for k, v in attributes.items() if
                 k not in ['variation_slots', 'race', 'age', 'gender', 'group_size']}

    for i in range(len(ordered_keys)):
        if incremental:
            # Incremental mode: cumulative subsets
            key_subset = ordered_keys[:i + 1]
            combo_key = ', '.join(key_subset)
        else:
            # Single mode: one key at a time as list
            key_subset = [ordered_keys[i]]
            combo_key = ordered_keys[i]

        combo_attr = dict(base_info)

        for k in key_subset:
            if k in top_attrs:
                combo_attr[k] = top_attrs[k]

        variation_variables = {}
        for k in key_subset:
            if k in var_attrs:
                variation_variables[k] = {'variable': [var_attrs[k]]}

        if variation_variables:
            original_slot = next(iter(var_slots.values()), None)
            start = original_slot.get('start', 0) if original_slot else 0
            end = original_slot.get('end', 0) if original_slot else 0

            combo_attr['variation_slots'] = {
                'slot1': {
                    'start': start,
                    'end': end,
                    'type': 'specific',
                    'variable': variation_variables
                }
            }

        # sentence = Refinement_attribute_person_T(combo_key, 1)
        bbox_number =2
        sentence = Refinement_attribute_person_T_VQA_Wh(attributes, key_subset, provide, T, bbox_number)
        result[combo_key] = sentence

    return result

def split_and_refine(attributes, bbox_number=1):
    base_info = {
        k: v for k, v in attributes.items()
        if k not in ['variation_slots', 'race', 'age', 'gender', 'group_size']
    }

    single_attribute_dicts = []

    # Handle top-level single attributes like race, age, gender, group_size
    for key in ['race', 'age', 'gender', 'group_size']:
        if key in attributes:
            new_attr = dict(base_info)
            new_attr[key] = attributes[key]
            new_attr['variation_slots'] = {}
            single_attribute_dicts.append(new_attr)

    # Handle each variable inside variation_slots
    var_slots = attributes.get('variation_slots', {})
    for slot_name, slot in var_slots.items():
        variables = slot.get('variable', {})
        for var_key, var_value in variables.items():
            new_attr = dict(base_info)
            new_slot = {
                slot_name: {
                    'start': slot.get('start'),
                    'end': slot.get('end'),
                    'type': slot.get('type', 'vague'),
                    'variable': {
                        var_key: var_value
                    }
                }
            }
            new_attr['variation_slots'] = new_slot
            single_attribute_dicts.append(new_attr)

    # Now call the refinement function on each
    results = []
    for attr in single_attribute_dicts:
        sentence = Refinement_attribute_person_T(attr, bbox_number)
        results.append(sentence)

    return results


def discretize_distances2(distances):
    # Define thresholds for discretization
    thresholds = [0.5, 1.5, 5, 10]
    labels = ["very close", "close", "moderate", "far"]

    # Discretize distances
    discretized = []
    # print(distances)
    for distance in distances:
        # print(distance)
        if distance is None:
            discretized.append("undefined")
            break
        elif distance <= thresholds[0]:
            discretized.append(labels[0])
        elif distance <= thresholds[1]:
            discretized.append(labels[1])
        elif distance <= thresholds[2]:
            discretized.append(labels[2])
        elif distance <= thresholds[3]:
            discretized.append(labels[3])
        else:
            discretized.append("very far")  # for distances exceeding the last threshold
    return discretized[0]

from collections import defaultdict

# def find_attributes_in_graph(selected_person, attributes_to_use, st_graph):
#     """
#     Refactored to perform a single graph query per frame and support generic attributes, including 'ID'.
#
#     Args:
#         selected_person (dict): Person data with variation_slots and other attributes.
#         attributes_to_use (list): List of attributes to extract (e.g., ['age', 'action', 'human', 'ID']).
#         st_graph: Spatio-temporal graph object with _search_graph method.
#
#     Returns:
#         dict: Mapping each slot name to a dict containing 'image_ids' and per-attribute lists of detections.
#     """
#     results = {}
#
#     for slot_name, slot_data in selected_person.get('variation_slots', {}).items():
#         start, end = slot_data['start'], slot_data['end']
#         frames = list(range(start, end + 1))
#         image_ids = [f"{frame:06d}.jpg" for frame in frames]
#
#         # Initialize storage
#         slot_dict = {'image_ids': image_ids}
#         for attr in attributes_to_use:
#             slot_dict[attr] = [[] for _ in frames]
#
#         # Determine which node attributes to request from the graph
#         # Always include bbox, ID, and type for filtering
#         node_attrs = ['bbox', 'ID', 'type']
#         # Add non-special attributes so graph returns them for filtering
#         extras = [a for a in attributes_to_use if a not in ('human', 'object', 'ID')]
#         node_attrs.extend(extras)
#
#         # Query once per frame
#         for idx, frame in enumerate(frames):
#             entries = st_graph._search_graph(
#                 {},                # no filters: get all nodes in this frame
#                 node_attrs,
#                 time_window=(frame, frame)
#             )
#
#             for e in entries:
#                 # Human/Object detection
#                 if 'human' in attributes_to_use and e.get('type') == 'human':
#                     slot_dict['human'][idx].append(e['bbox'])
#                 if 'object' in attributes_to_use and e.get('type') == 'object':
#                     slot_dict['object'][idx].append(e['bbox'])
#
#                 # Other attribute-based detections
#                 for attr in extras:
#                     # Gather possible values for this attr
#                     if attr in slot_data.get('variable', {}):
#                         vals = slot_data['variable'][attr]['variable']
#                     else:
#                         vals = [selected_person.get(attr)] if selected_person.get(attr) is not None else []
#
#                     if e.get(attr) in vals:
#                         slot_dict[attr][idx].append(e['bbox'])
#
#                 # ID extraction
#                 if 'ID' in attributes_to_use and 'ID' in e:
#                     slot_dict['ID'][idx].append(e['ID'])
#
#         results[slot_name] = slot_dict
#
#     return results
# def find_attributes_in_graph(selected_person, attributes_to_use, st_graph):
#     """
#     For each variation‐slot in selected_person, query st_graph once per frame
#     and collect BOTH bbox and ID for:
#       • each individual attribute in attributes_to_use (age, action, human)
#       • nodes matching *all* of those attributes simultaneously
#
#     Returns:
#         dict of slot_name → {
#             "image_ids": [...],
#             "ID":    [ [...], … ],
#             "bbox":  [ […], … ],
#             "<attr>": {"bbox": [ … ], "ID": [ … ]},  for attr in ['age','action','human']
#         }
#     """
#     results = {}
#
#     # We treat 'ID' as purely an output‐collector, so filter only on the other attrs:
#     filter_attrs = [a for a in attributes_to_use if a != 'node_name']
#     # among those, 'human' is special (checks e['type']=='human'), the rest we pull from slot_data.variable
#     extras = [a for a in filter_attrs if a != 'human']
#
#     for slot_name, slot_data in selected_person.get('variation_slots', {}).items():
#         start, end = slot_data['start'], slot_data['end']
#         frames = list(range(start, end + 1))
#         image_ids = [f"{f:06d}.jpg" for f in frames]
#
#         # initialize per‐frame lists
#         slot_dict = {
#             "image_ids": image_ids,
#             "ID":   [[] for _ in frames],
#             "bbox": [[] for _ in frames],
#         }
#         # one sub‐dict per attribute to collect its bbox+ID
#         for attr in filter_attrs:
#             slot_dict[attr] = {
#                 "bbox": [[] for _ in frames],
#                 "ID":   [[] for _ in frames],
#             }
#
#         # ask the graph to return bbox, ID, type, plus any extras so we can filter on them:
#         node_attrs = ['bbox', 'node_name', 'type'] + extras
#
#         for idx, frame in enumerate(frames):
#             entries = st_graph._search_graph(
#                 {},                # no additional filters
#                 node_attrs,
#                 time_window=(frame, frame)
#             )
#
#             for e in entries:
#                 # --- check each individual attribute ---
#                 matches_all = True
#
#                 # 1) human?
#                 if 'human' in filter_attrs:
#                     is_h = (e.get('type') == 'human')
#                     if is_h:
#                         slot_dict['human']['bbox'][idx].append(e['bbox'])
#                         slot_dict['human']['ID'][idx].append(e['node_name'])
#                     else:
#                         matches_all = False
#
#                 # 2) other extras (age, action, …)
#                 for attr in extras:
#                     # what values count this slot?
#                     vals = slot_data.get('variable', {}) \
#                                    .get(attr, {}) \
#                                    .get('variable',
#                                         [selected_person.get(attr)])
#                     if e.get(attr) in vals:
#                         slot_dict[attr]['bbox'][idx].append(e['bbox'])
#                         slot_dict[attr]['ID'][idx].append(e['node_name'])
#                     else:
#                         matches_all = False
#
#                 # --- if this entry matched *all* requested attrs, collect it top‐level ---
#                 if matches_all:
#                     slot_dict['bbox'][idx].append(e['bbox'])
#                     slot_dict['ID'][idx].append(e['node_name'])
#
#         results[slot_name] = slot_dict
#
#     return results
# def find_attributes_in_graph(selected_person, attributes_to_use, st_graph):
#     """
#     For each variation-slot in selected_person, query st_graph once per frame
#     and collect BOTH bbox and ID for:
#       • each individual attribute in attributes_to_use (e.g., age, action, human)
#       • nodes matching *all* of those attributes simultaneously (combined)
#       • for pair-level tasks (hhi, hhg, hoi), collect ID/bbox of main and pair
#
#     Returns:
#         dict of slot_name → {
#             "image_ids": [...],
#             "final_ID":    [ [...], … ],
#             "final_bbox":  [ […], … ],
#             "<attr>": {
#                 "target_bbox": [ … ], "target_ID": [ … ]
#             } or for interaction types:
#             "<interaction_attr>": {
#                 "target_bbox": [ … ],
#                 "target_ID":   [ … ],
#                 "pair_bbox":   [ … ],
#                 "pair_ID":     [ … ]
#             }
#         }
#     """
#     results = {}
#
#     filter_attrs = [a for a in attributes_to_use if a != 'node_name']
#     extras = [a for a in filter_attrs if a != 'human']
#
#     interaction_attrs = ['hhi', 'hhg', 'hoi']
#
#     for slot_name, slot_data in selected_person.get('variation_slots', {}).items():
#         start, end = slot_data['start'], slot_data['end']
#         frames = list(range(start, end + 1))
#         image_ids = [f"{f:06d}.jpg" for f in frames]
#
#         slot_dict = {
#             "image_ids": image_ids,
#             "final_ID":   [[] for _ in frames],
#             "final_bbox": [[] for _ in frames]
#         }
#
#         for attr in filter_attrs:
#             if attr in interaction_attrs:
#                 slot_dict[attr] = {
#                     "target_bbox": [[] for _ in frames],
#                     "target_ID":   [[] for _ in frames],
#                     "pair_bbox":   [[] for _ in frames],
#                     "pair_ID":     [[] for _ in frames]
#                 }
#             else:
#                 slot_dict[attr] = {
#                     "target_bbox": [[] for _ in frames],
#                     "target_ID":   [[] for _ in frames]
#                 }
#
#         node_attrs = ['bbox', 'type', 'node_name'] + extras
#
#         for idx, frame in enumerate(frames):
#             entries = st_graph._search_graph({}, node_attrs, time_window=(frame, frame))
#
#             for e in entries:
#                 matches_all = True
#
#                 if 'human' in filter_attrs:
#                     if e.get('type') == 'human':
#                         slot_dict['human']['target_bbox'][idx].append(e['bbox'])
#                         slot_dict['human']['target_ID'][idx].append(e['node_name'])
#                     else:
#                         matches_all = False
#
#                 for attr in extras:
#                     vals = (slot_data.get('variable', {})
#                                      .get(attr, {})
#                                      .get('variable', [selected_person.get(attr)]))
#
#                     if attr in interaction_attrs:
#                         if e.get(attr) in vals:
#                             slot_dict[attr]['target_bbox'][idx].append(e.get('bbox'))
#                             slot_dict[attr]['target_ID'][idx].append(e.get('node_name'))
#                             slot_dict[attr]['pair_bbox'][idx].append(e.get('pair_bbox', None))
#                             slot_dict[attr]['pair_ID'][idx].append(e.get('pair_id', None))
#                         else:
#                             matches_all = False
#                     else:
#                         if e.get(attr) in vals:
#                             slot_dict[attr]['target_bbox'][idx].append(e['bbox'])
#                             slot_dict[attr]['target_ID'][idx].append(e['node_name'])
#                         else:
#                             matches_all = False
#
#                 if matches_all:
#                     slot_dict['final_bbox'][idx].append(e['bbox'])
#                     slot_dict['final_ID'][idx].append(e['node_name'])
#
#         results[slot_name] = slot_dict
#
#     return results
# def find_attributes_in_graph(selected_person, attributes_to_use, st_graph):
#     results = {}
#     person_id = selected_person["id"]
#     variation_slots = selected_person["variation_slots"]
#
#     def get_empty_structure(attr):
#         if attr in ["age", "action", "human"]:
#             return {"target_bbox": [], "target_ID": []}
#         elif attr in ["hhg", "hhi"]:
#             return {
#                 "target_bbox": [], "target_ID": [],
#                 "pair_bbox": [], "pair_ID": [],
#                 "pair_gender": [], "pair_age": [], "pair_race": [], "pair_slot_id": []
#             }
#         elif attr == "hoi":
#             return {
#                 "target_bbox": [], "target_ID": [],
#                 "pair_bbox": [], "pair_ID": [],
#                 "pair_gender": [], "pair_age": [], "pair_race": [], "pair_slot_id": [],
#                 "obj_name": [], "tracking_id": [], "category_id": [], "id": []
#             }
#         else:
#             return {}
#
#     for slot_name, slot_data in variation_slots.items():
#         start = slot_data["start"]
#         end = slot_data["end"]
#         variable = slot_data["variable"]
#
#         results[slot_name] = {
#             "image_ids": [],
#             "final_ID": [],
#             "final_bbox": []
#         }
#
#         for attr in attributes_to_use:
#             results[slot_name][attr] = get_empty_structure(attr)
#
#         for frame in range(start, end + 1):
#             results[slot_name]["image_ids"].append(f"{frame:06d}.jpg")
#
#             for attr in attributes_to_use:
#                 if attr == "age":
#                     query = {"age": selected_person.get("age")}
#                     fields = ["id", "bbox"]
#                 elif attr == "human":
#                     query = {}
#                     fields = ["id", "bbox"]
#                 elif attr == "action":
#                     action_value = variable.get("action", {}).get("variable", [None])[0]
#                     query = {"action": action_value}
#                     fields = ["id", "bbox"]
#                 elif attr == "hhg":
#                     hhg_info = variable.get("hhg", {})
#                     query = {
#                         # "pair_id": hhg_info.get("id_pair"),
#                         "interactions": hhg_info.get("variable", [None])[0]
#                     }
#                     fields = ["id", "bbox", "pair_id", "pair_bbox"]
#                 elif attr == "hhi":
#                     hhi_info = variable.get("hhi", {})
#                     query = {
#                         # "pair_id": hhi_info.get("id_pair"),
#                         "interactions": hhi_info.get("variable", [None])[0]
#                     }
#                     fields = ["id", "bbox", "pair_id", "pair_bbox"]
#                 elif attr == "hoi":
#                     hoi_info = variable.get("hoi", {})
#                     query = {
#                         "obj_name": hoi_info.get("obj_name"),
#                         "interactions": hoi_info.get("variable", [None])[0]
#                     }
#                     fields = ["id", "bbox", "obj_id", "obj_bbox"]
#                 else:
#                     continue
#
#                 matches = st_graph._search_graph(query, fields, (frame, frame))
#
#                 for match in matches:
#                     if attr in ["age", "human", "action"]:
#                         results[slot_name][attr]["target_ID"].append(match.get("id"))
#                         results[slot_name][attr]["target_bbox"].append(match.get("bbox"))
#
#                     elif attr in ["hhg", "hhi"]:
#                         results[slot_name][attr]["target_ID"].append(match.get("id"))
#                         results[slot_name][attr]["target_bbox"].append(match.get("bbox"))
#                         results[slot_name][attr]["pair_ID"].append(match.get("pair_id"))
#                         results[slot_name][attr]["pair_bbox"].append(match.get("pair_bbox"))
#
#                         pair_info = variable[attr]
#                         results[slot_name][attr]["pair_gender"].append(pair_info.get("gender_pair"))
#                         results[slot_name][attr]["pair_age"].append(pair_info.get("age_pair"))
#                         results[slot_name][attr]["pair_race"].append(pair_info.get("race_pair"))
#                         results[slot_name][attr]["pair_slot_id"].append(pair_info.get("id_pair"))
#
#                     elif attr == "hoi":
#                         results[slot_name][attr]["target_ID"].append(match.get("id"))
#                         results[slot_name][attr]["target_bbox"].append(match.get("bbox"))
#                         results[slot_name][attr]["pair_ID"].append(match.get("obj_id"))
#                         results[slot_name][attr]["pair_bbox"].append(match.get("obj_bbox"))
#
#                         obj_info = variable[attr]
#                         results[slot_name][attr]["pair_gender"].append(None)
#                         results[slot_name][attr]["pair_age"].append(None)
#                         results[slot_name][attr]["pair_race"].append(None)
#                         results[slot_name][attr]["pair_slot_id"].append(None)
#
#                         results[slot_name][attr]["obj_name"].append(obj_info.get("obj_name"))
#                         results[slot_name][attr]["tracking_id"].append(obj_info.get("tracking_id"))
#                         results[slot_name][attr]["category_id"].append(obj_info.get("category_id"))
#                         results[slot_name][attr]["id"].append(obj_info.get("id"))
#
#     return results

# def find_attributes_in_graph(selected_person, attributes_to_use, st_graph):
#     results = {}
#     variation_slots = selected_person["variation_slots"]
#
#     def get_empty_structure(attr):
#         if attr in ["age","race","gender","distance","SR_Robot_Ref", "action", "human"]:
#             return {"target_bbox": [], "target_ID": []}
#         elif attr in ["hhg", "hhi"]:
#             return {
#                 "target_bbox": [], "target_ID": [],
#                 "pair_bbox": [], "pair_ID": [],
#                 "pair_gender": [], "pair_age": [], "pair_race": [], "pair_slot_id": []
#             }
#         elif attr == "hoi":
#             return {
#                 "target_bbox": [], "target_ID": [],
#                 "pair_bbox": [], "pair_ID": [],
#                 # "pair_gender": [], "pair_age": [], "pair_race": [], "pair_slot_id": [],
#                 "obj_name": [], "tracking_id": [], "category_id": [], "id": []
#             }
#         else:
#             return {}
#
#     for slot_name, slot_data in variation_slots.items():
#         start, end = slot_data["start"], slot_data["end"]
#         variable = slot_data["variable"]
#
#         # results[slot_name] = {"image_ids": [], "final_ID": [], "final_bbox": []}
#         if any(key in ["hhi", "hhg", "hoi"] for key in variable.keys()):
#             results[slot_name] = {
#                 "image_ids": [],
#                 "final_ID": [], "final_bbox": [],
#                 "final_ID_pair": [], "final_bbox_pair": []
#             }
#         else:
#             results[slot_name] = {
#                 "image_ids": [],
#                 "final_ID": [], "final_bbox": []
#             }
#
#         for attr in attributes_to_use:
#             results[slot_name][attr] = get_empty_structure(attr)
#
#         for frame in range(start, end + 1):
#             results[slot_name]["image_ids"].append(f"{frame:06d}.jpg")
#
#             for attr in attributes_to_use:
#                 # Base fields
#                 fields = ["id", "bbox"]
#                 # Start building query
#                 # query = {"type": attr}
#                 query = {"type": "human"}  # got an issue
#
#                 if attr == "age":
#                     query.update({"age": selected_person.get("age")})
#                 elif attr == "race":
#                     query.update({"race": selected_person.get("race")})
#                 elif attr == "gender":
#                     query.update({"gender": selected_person.get("gender")})
#                 elif attr == "distance":
#                     info = variable.get("distance", {})
#                     query.update({"distance": info.get("variable", [None])[0]})
#                 elif attr == "SR_Robot_Ref":
#                     info = variable.get("SR_Robot_Ref", {})
#                     query.update({"SR_Robot_Ref": info.get("variable", [None])[0]})
#                 elif attr == "human":
#                     pass  # no additional filters
#                 elif attr == "action":
#                     action_value = variable.get("action", {}).get("variable", [None])[0]
#                     query.update({"action": action_value})
#                 elif attr in ["hhg", "hhi"]:
#                     info = variable.get(attr, {})
#                     query.update({"interactions": info.get("variable", [None])[0]})
#                     # include pair filters if available
#                     # if info.get("id_pair") is not None:
#                     #     query.update({"pair_id": info.get("id_pair")})
#                     if info.get("gender_pair") is not None:
#                         query.update({"gender_pair": info.get("gender_pair")})
#                     if info.get("age_pair") is not None:
#                         query.update({"age_pair": info.get("age_pair")})
#                     if info.get("race_pair") is not None:
#                         query.update({"race_pair": info.get("race_pair")})
#                     fields += ["id_pair", "bbox_pair"]
#                 elif attr == "hoi":
#                     info = variable.get("hoi", {})
#                     query.update({
#                         "obj_name": info.get("obj_name"),
#                         "interactions": info.get("variable", [None])[0],
#                         "category_id": info.get("category_id")
#                     })
#                     fields = ["id", "bbox", "obj_name", "category_id"]
#
#                 # Perform search
#                 matches = st_graph._search_graph(query, fields, (frame, frame))
#
#                 for match in matches:
#                     # Common target entries
#                     # print(match)
#                     # print(results)
#                     results[slot_name][attr]["target_ID"].append(match.get("id"))
#                     results[slot_name][attr]["target_bbox"].append(match.get("bbox"))
#
#                     if attr in ["hhg", "hhi"]:
#                         results[slot_name][attr]["pair_ID"].append(match.get("id_pair"))
#                         results[slot_name][attr]["pair_bbox"].append(match.get("bbox_pair"))
#                         pair_info = variable[attr]
#                         results[slot_name][attr]["pair_gender"].append(pair_info.get("gender_pair"))
#                         results[slot_name][attr]["pair_age"].append(pair_info.get("age_pair"))
#                         results[slot_name][attr]["pair_race"].append(pair_info.get("race_pair"))
#                         # results[slot_name][attr]["pair_slot_id"].append(pair_info.get("id_pair"))
#
#                     elif attr == "hoi":
#                         results[slot_name][attr]["pair_ID"].append(match.get("obj_id"))
#                         results[slot_name][attr]["pair_bbox"].append(match.get("obj_bbox"))
#                         # results[slot_name][attr]["pair_gender"].append(None)
#                         # results[slot_name][attr]["pair_age"].append(None)
#                         # results[slot_name][attr]["pair_race"].append(None)
#                         # results[slot_name][attr]["pair_slot_id"].append(None)
#                         info = variable.get("hoi", {})
#                         results[slot_name][attr]["obj_name"].append(info.get("obj_name"))
#                         results[slot_name][attr]["tracking_id"].append(info.get("tracking_id"))
#                         results[slot_name][attr]["category_id"].append(info.get("category_id"))
#                         results[slot_name][attr]["id"].append(info.get("id"))
#     results = fill_all_slots(results)
#     results = get_final_bbox(results)
#
#     return results

def find_attributes_in_graph(selected_person, attributes_to_use, st_graph):
    results = {}
    variation_slots = selected_person["variation_slots"]

    def get_empty_structure(attr):
        if attr in ["age","race","gender","distance","SR_Robot_Ref", "action", "human"]:
            return {"target_bbox": [], "target_ID": []}
        elif attr in ["hhg", "hhi"]:
            return {
                "target_bbox": [], "target_ID": [],
                "pair_bbox": [], "pair_ID": [],
                "pair_gender": [], "pair_age": [], "pair_race": [], "pair_slot_id": []
            }
        elif attr == "hoi":
            return {
                "target_bbox": [], "target_ID": [],
                "pair_bbox": [], "pair_ID": [],
                "obj_name": [], "tracking_id": [], "category_id": [], "id": []
            }
        else:
            return {}

    # which fields we want back per attr
    FIELDS_BY_ATTR = {
        "default": ["id", "bbox"],
        "hhi": ["id", "bbox", "id_pair", "bbox_pair"],
        "hhg": ["id", "bbox", "id_pair", "bbox_pair"],
        "hoi": ["id", "bbox", "obj_id", "obj_bbox", "obj_name", "category_id"],
    }

    for slot_name, slot_data in variation_slots.items():
        start, end = slot_data["start"], slot_data["end"]
        variable = slot_data["variable"]

        if any(k in ["hhi", "hhg", "hoi"] for k in variable.keys()):
            results[slot_name] = {
                "image_ids": [],
                "final_ID": [], "final_bbox": [],
                "final_ID_pair": [], "final_bbox_pair": []
            }
        else:
            results[slot_name] = {
                "image_ids": [],
                "final_ID": [], "final_bbox": []
            }

        for attr in attributes_to_use:
            results[slot_name][attr] = get_empty_structure(attr)

        for frame in range(start, end + 1):
            results[slot_name]["image_ids"].append(f"{frame:06d}.jpg")

            # for attr in attributes_to_use:
            #     gtype = TYPE_BY_ATTR.get(attr, "human")
            #     fields = FIELDS_BY_ATTR.get(attr, FIELDS_BY_ATTR["default"]).copy()
            #     # Build query
            #     query = {"node_type": gtype}
            #
            #     if attr in ["age","race","gender"]:
            #         query["asked_atrr"][attr] = selected_person.get(attr)
            #     elif attr in ["distance","SR_Robot_Ref","action"]:
            #         info = variable.get(attr, {})
            #         query["asked_atrr"][attr] = (info.get("variable") or [None])[0]
            #     elif attr in ["hhg", "hhi"]:
            #         info = variable.get(attr, {})
            #         query["asked_atrr"][attr]["interactions"] = (info.get("variable") or [None])[0]
            #         if info.get("gender_pair") is not None:
            #             query["asked_atrr"][attr]["gender_pair"] = info.get("gender_pair")
            #         if info.get("age_pair") is not None:
            #             query["asked_atrr"][attr]["age_pair"] = info.get("age_pair")
            #         if info.get("race_pair") is not None:
            #             query["asked_atrr"][attr]["race_pair"] = info.get("race_pair")
            #     elif attr == "hoi":
            #         info = variable.get("hoi", {})
            #         query["asked_atrr"][attr].update({
            #             "obj_name": info.get("obj_name"),
            #             "interactions": (info.get("variable") or [None])[0],
            #             "category_id": info.get("category_id"),
            #         })
            #
            #     # if attr == "age":
            #     #     query["age"] = selected_person.get("age")
            #     # elif attr == "race":
            #     #     query["race"] = selected_person.get("race")
            #     # elif attr == "gender":
            #     #     query["gender"] = selected_person.get("gender")
            #     # elif attr == "distance":
            #     #     info = variable.get("distance", {})
            #     #     query["distance"] = (info.get("variable") or [None])[0]
            #     # elif attr == "SR_Robot_Ref":
            #     #     info = variable.get("SR_Robot_Ref", {})
            #     #     query["SR_Robot_Ref"] = (info.get("variable") or [None])[0]
            #     # elif attr == "action":
            #     #     action_value = (variable.get("action", {}).get("variable") or [None])[0]
            #     #     query["action"] = action_value
            #     # elif attr in ["hhg", "hhi"]:
            #     #     info = variable.get(attr, {})
            #     #     query["interactions"] = (info.get("variable") or [None])[0]
            #     #     if info.get("gender_pair") is not None:
            #     #         query["gender_pair"] = info.get("gender_pair")
            #     #     if info.get("age_pair") is not None:
            #     #         query["age_pair"] = info.get("age_pair")
            #     #     if info.get("race_pair") is not None:
            #     #         query["race_pair"] = info.get("race_pair")
            #     # elif attr == "hoi":
            #     #     info = variable.get("hoi", {})
            #     #     query.update({
            #     #         "obj_name": info.get("obj_name"),
            #     #         "interactions": (info.get("variable") or [None])[0],
            #     #         "category_id": info.get("category_id"),
            #     #     })
            #
            #     # Search
            #     matches = st_graph._search_graph(query, fields, (frame, frame))
            #
            #     # Collect
            #     for match in matches:
            #         results[slot_name][attr]["target_ID"].append(match.get("id"))
            #         results[slot_name][attr]["target_bbox"].append(match.get("bbox"))
            #
            #         if attr in ["hhg", "hhi"]:
            #             results[slot_name][attr]["pair_ID"].append(match.get("id_pair"))
            #             results[slot_name][attr]["pair_bbox"].append(match.get("bbox_pair"))
            #             pair_info = variable.get(attr, {})
            #             results[slot_name][attr]["pair_gender"].append(pair_info.get("gender_pair"))
            #             results[slot_name][attr]["pair_age"].append(pair_info.get("age_pair"))
            #             results[slot_name][attr]["pair_race"].append(pair_info.get("race_pair"))
            #
            #         elif attr == "hoi":
            #             # prefer fields from match; fall back to slot variable
            #             info = variable.get("hoi", {})
            #             results[slot_name][attr]["pair_ID"].append(match.get("obj_id"))
            #             results[slot_name][attr]["pair_bbox"].append(match.get("obj_bbox"))
            #             results[slot_name][attr]["obj_name"].append(match.get("obj_name") or info.get("obj_name"))
            #             results[slot_name][attr]["category_id"].append(match.get("category_id") or info.get("category_id"))
            #             results[slot_name][attr]["tracking_id"].append(info.get("tracking_id"))
            #             results[slot_name][attr]["id"].append(info.get("id"))
            for attr in attributes_to_use:
                if "human" in attributes_to_use:
                    gtype = "human"
                elif "object" in attributes_to_use:
                    gtype = "object"
                else:
                    gtype = None  # or a default value if needed

                fields = FIELDS_BY_ATTR.get(attr, FIELDS_BY_ATTR["default"]).copy()

                # Build query
                query = {
                    "node_type": gtype,
                    "asked_attributes": {},  # holds values per-attr (dict)
                    # "asked_attributes_name": attr,  # optional: the full list of names
                }

                if attr in ["age", "race", "gender"]:
                    query["asked_attributes"][attr] = selected_person.get(attr)

                elif attr in ["distance", "SR_Robot_Ref", "action"]:
                    info = variable.get(attr, {})
                    query["asked_attributes"][attr] = (info.get("variable") or [None])[0]

                elif attr in ["hhg", "hhi"]:
                    info = variable.get(attr, {})
                    aa = query["asked_attributes"].setdefault(attr, {})
                    aa["interactions"] = (info.get("variable") or [None])[0]
                    if info.get("gender_pair") is not None:
                        aa["gender_pair"] = info.get("gender_pair")
                    if info.get("age_pair") is not None:
                        aa["age_pair"] = info.get("age_pair")
                    if info.get("race_pair") is not None:
                        aa["race_pair"] = info.get("race_pair")

                elif attr == "hoi":
                    info = variable.get("hoi", {})
                    aa = query["asked_attributes"].setdefault(attr, {})
                    aa.update({
                        "obj_name": info.get("obj_name"),
                        "interactions": (info.get("variable") or [None])[0],
                        "category_id": info.get("category_id"),
                    })

                # now use `query` and `fields`:
                matches = st_graph._search_graph(query, fields, (frame, frame))
                for match in matches:
                    # Common target entries
                    # print(match)
                    # print(results)
                    results[slot_name][attr]["target_ID"].append(match.get("id"))
                    results[slot_name][attr]["target_bbox"].append(match.get("bbox"))

                    if attr in ["hhg", "hhi"]:
                        results[slot_name][attr]["pair_ID"].append(match.get("id_pair"))
                        results[slot_name][attr]["pair_bbox"].append(match.get("bbox_pair"))
                        pair_info = variable[attr]
                        results[slot_name][attr]["pair_gender"].append(pair_info.get("gender_pair"))
                        results[slot_name][attr]["pair_age"].append(pair_info.get("age_pair"))
                        results[slot_name][attr]["pair_race"].append(pair_info.get("race_pair"))
                        # results[slot_name][attr]["pair_slot_id"].append(pair_info.get("id_pair"))

                    elif attr == "hoi":
                        results[slot_name][attr]["pair_ID"].append(match.get("obj_id"))
                        results[slot_name][attr]["pair_bbox"].append(match.get("obj_bbox"))
                        # results[slot_name][attr]["pair_gender"].append(None)
                        # results[slot_name][attr]["pair_age"].append(None)
                        # results[slot_name][attr]["pair_race"].append(None)
                        # results[slot_name][attr]["pair_slot_id"].append(None)
                        info = variable.get("hoi", {})
                        results[slot_name][attr]["obj_name"].append(info.get("obj_name"))
                        results[slot_name][attr]["tracking_id"].append(info.get("tracking_id"))
                        results[slot_name][attr]["category_id"].append(info.get("category_id"))
                        results[slot_name][attr]["id"].append(info.get("id"))

    results = fill_all_slots(results)
    results = get_final_bbox(results)
    return results


def find_attributes_in_graph_VQA(selected_person, ask, attributes_to_use, st_graph):
    # provide = attributes_to_use
    results = {}
    variation_slots = selected_person["variation_slots"]

    def get_empty_structure(attr):
        if attr in ["age","race","gender","distance","SR_Robot_Ref", "action", "human"]:
            return {"target_answer": [], "target_ID": []}
        elif attr in ["hhg", "hhi"]:
            return {
                "target_answer": [], "target_ID": [],
                # "pair_bbox": [], "pair_ID": [],
                # "pair_gender": [], "pair_age": [], "pair_race": [], "pair_slot_id": []
            }
        elif attr == "hoi":
            return {
                "target_answer": [], "target_ID": [],
                # "pair_bbox": [], "pair_ID": [],
                # "pair_gender": [], "pair_age": [], "pair_race": [], "pair_slot_id": [],
                # "obj_name": [], "tracking_id": [], "category_id": [], "id": []
            }
        else:
            return {}

    for slot_name, slot_data in variation_slots.items():
        start, end = slot_data["start"], slot_data["end"]
        variable = slot_data["variable"]

        # results[slot_name] = {"image_ids": [], "final_ID": [], "final_bbox": []}
        if any(key in ["hhi", "hhg", "hoi"] for key in variable.keys()):
            results[slot_name] = {
                "image_ids": [],
                "final_ID": [], "final_answer": [],
                # "final_ID_pair": [], "final_bbox_pair": []
            }
        else:
            results[slot_name] = {
                "image_ids": [],
                "final_ID": [], "final_answer": []
            }

        for attr in ask:
            results[slot_name][attr] = get_empty_structure(attr)

        for frame in range(start, end + 1):
            results[slot_name]["image_ids"].append(f"{frame:06d}.jpg")

            for attr in attributes_to_use:
                # Base fields
                # fields = ["id", "bbox"]
                fields = ask
                # Start building query
                query = {"type": attr}

                if attr == "age":
                    query.update({"age": selected_person.get("age")})
                elif attr == "race":
                    query.update({"race": selected_person.get("race")})
                elif attr == "gender":
                    query.update({"gender": selected_person.get("gender")})
                elif attr == "distance":
                    info = variable.get("distance", {})
                    query.update({"distance": info.get("variable", [None])[0]})
                elif attr == "SR_Robot_Ref":
                    info = variable.get("SR_Robot_Ref", {})
                    query.update({"SR_Robot_Ref": info.get("variable", [None])[0]})
                elif attr == "human":
                    pass  # no additional filters
                elif attr == "action":
                    action_value = variable.get("action", {}).get("variable", [None])[0]
                    query.update({"action": action_value})
                elif attr in ["hhg", "hhi"]:
                    info = variable.get(attr, {})
                    query.update({"interactions": info.get("variable", [None])[0]})
                    # include pair filters if available
                    # if info.get("id_pair") is not None:
                    #     query.update({"pair_id": info.get("id_pair")})
                    if info.get("gender_pair") is not None:
                        query.update({"gender_pair": info.get("gender_pair")})
                    if info.get("age_pair") is not None:
                        query.update({"age_pair": info.get("age_pair")})
                    if info.get("race_pair") is not None:
                        query.update({"race_pair": info.get("race_pair")})
                    # fields += ["id_pair", "bbox_pair"]
                elif attr == "hoi":
                    info = variable.get("hoi", {})
                    query.update({
                        "obj_name": info.get("obj_name"),
                        "interactions": info.get("variable", [None])[0],
                        "category_id": info.get("category_id")
                    })
                    # fields = ["id", "bbox", "obj_name", "category_id"]

                # Perform search
                matches = st_graph._search_graph(query, fields, (frame, frame))

                for match in matches:
                    # Common target entries
                    # print(match)
                    # print(results)

                    for asked in fields:
                        if match.get("id") is not None and match.get("id") not in results[slot_name][asked][
                            "target_ID"]:
                            results[slot_name][asked]["target_ID"].append(match.get("id"))
                        if match.get(asked) is not None and match.get(asked) not in results[slot_name][asked][
                            "target_answer"]:
                            results[slot_name][asked]["target_answer"].append(match.get(asked))
                        if asked in ["hhg", "hhi"]:
                            print('i')
                            # results[slot_name][asked]["pair_ID"].append(match.get("id_pair"))
                            # results[slot_name][attr]["pair_bbox"].append(match.get("bbox_pair"))
                            # pair_info = variable[attr]
                            # results[slot_name][attr]["pair_gender"].append(pair_info.get("gender_pair"))
                            # results[slot_name][attr]["pair_age"].append(pair_info.get("age_pair"))
                            # results[slot_name][attr]["pair_race"].append(pair_info.get("race_pair"))
                            # results[slot_name][attr]["pair_slot_id"].append(pair_info.get("id_pair"))

                        elif attr == "hoi":
                            results[slot_name][asked]["pair_ID"].append(match.get("obj_id"))
                            # results[slot_name][attr]["pair_bbox"].append(match.get("obj_bbox"))
                            # results[slot_name][attr]["pair_gender"].append(None)
                            # results[slot_name][attr]["pair_age"].append(None)
                            # results[slot_name][attr]["pair_race"].append(None)
                            # results[slot_name][attr]["pair_slot_id"].append(None)
                            info = variable.get("hoi", {})
                            # results[slot_name][attr]["obj_name"].append(info.get("obj_name"))
                            # results[slot_name][attr]["tracking_id"].append(info.get("tracking_id"))
                            # results[slot_name][attr]["category_id"].append(info.get("category_id"))
                            # results[slot_name][attr]["id"].append(info.get("id"))
    results = fill_all_slots_VQA(results)
    # results = get_final_bbox(results)

    return results
def get_random_permutation(options):
    options = list(set(options))  # Ensure options is a list of unique elements
    random.shuffle(options)  # Shuffle the options in place
    return tuple(options)  # Convert the shuffled list to a tuple

def generate_options(answer, feature_to_ask, max_length=15):
    """
    Generate a list of options starting with the answer and ensuring all answer items are included.

    Args:
        answer (list): A list of correct answers.
        feature_to_ask (list): Features to use for generating additional options.
        max_length (int): Maximum length of the options list. Default is 15.

    Returns:
        list: A list of options of specified maximum length, including all items from the answer.
    """

    # Ensure the answer list has unique items
    # print(answer)
    answer = list(set(answer))
    # Initialize options with items from answer
    options = answer.copy()

    # Add additional options from the specified features
    for A_feature in feature_to_ask:
        if len(options) >= max_length:
            break
        options_to_add = Options.get(A_feature, [])
        options.extend(options_to_add)

    # Ensure options contain all items from the answer and are exactly max_length
    if len(options) > max_length:
        options = options[:max_length - len(answer)] + answer
    options = options[:max_length]
    options = get_random_permutation(options)

    return list(options)
def get_final_bbox(results):
    for slot_data in results.values():
        final_bbox = slot_data.get('final_bbox')
        return None if final_bbox == [] else results
def update_final_structures_by_results(final_structures: dict, results: dict) -> dict:
    from collections import defaultdict

    updated_structures = {}

    for slot, data in final_structures.items():
        updated_slot = {
            'image_ids': data.get('image_ids', []),
            'final_ID': data.get('final_ID', []),
            'final_bbox': data.get('final_bbox', []),
            'final_ID_pair': data.get('final_ID_pair', []),
            'final_bbox_pair': data.get('final_bbox_pair', [])
        }

        for key_combo in results:
            keys = [k.strip() for k in key_combo.split(',')]
            id_sets = []
            bbox_dicts = []

            # Collect ID sets and bbox lookup for each attribute in the combo
            for k in keys:
                if k in data and 'target_ID' in data[k] and 'target_bbox' in data[k]:
                    id_sets.append(set(data[k]['target_ID']))
                    bbox_dicts.append({id_: bbox for id_, bbox in zip(data[k]['target_ID'], data[k]['target_bbox'])})

            if id_sets:
                # Find common IDs across all attributes
                common_ids = set.intersection(*id_sets)
                final_bboxes = []
                final_ids = []

                if common_ids:
                    # Use the first bbox_dict to pull bounding boxes (they should match across all anyway)
                    first_bbox_dict = bbox_dicts[0]
                    for pid in common_ids:
                        if pid in first_bbox_dict:
                            final_ids.append(pid)
                            final_bboxes.append(first_bbox_dict[pid])

                # Add to updated structure
                updated_slot[key_combo] = {
                    'target_ID': final_ids,
                    'target_bbox': final_bboxes
                }

        updated_structures[slot] = updated_slot

    return updated_structures

def update_final_structures_by_results_VQA(final_structures: dict, results: dict) -> dict:
    updated_structures = {}

    for slot, data in final_structures.items():
        updated_slot = {
            'image_ids': data.get('image_ids', []),
            'final_ID': data.get('final_ID', []),
            'final_answer': data.get('final_answer', [])
        }

        for key_combo in results:
            keys = [k.strip() for k in key_combo.split(',')]
            id_sets = []
            answer_dicts = []

            # Collect ID sets and answer lookups for each attribute
            for k in keys:
                if k in data and 'target_ID' in data[k] and 'target_answer' in data[k]:
                    id_sets.append(set(data[k]['target_ID']))
                    answer_dicts.append({id_: ans for id_, ans in zip(data[k]['target_ID'], data[k]['target_answer'])})

            if id_sets:
                # Find common IDs
                common_ids = set.intersection(*id_sets)
                final_ids = []
                final_answers = []

                for pid in common_ids:
                    final_ids.append(pid)
                    for d in answer_dicts:
                        if pid in d:
                            final_answers.append(d[pid])  # flat list

                updated_slot[key_combo] = {
                    'target_ID': final_ids,
                    'target_answer': final_answers  # flat list
                }

        updated_structures[slot] = updated_slot

    return updated_structures


# def fill_all_slots(data):
#     pair_keys = {'hhi', 'hhg', 'hoi'}
#
#     for slot_key, slot in data.items():
#         target_ID_sets = []
#         target_bbox_sets = []
#         pair_ID_sets = []
#         pair_bbox_sets = []
#         has_pair_data = False  # Flag to track presence of hhi/hhg/hoi
#
#         for subkey, subdict in slot.items():
#             if isinstance(subdict, dict):
#                 # Target ID and BBox
#                 target_ids = subdict.get('target_ID', [])
#                 target_bboxes = subdict.get('target_bbox', [])
#                 if target_ids:
#                     target_ID_sets.append(set(target_ids))
#                 if target_bboxes:
#                     target_bbox_sets.append(set(tuple(b) for b in target_bboxes))
#
#                 # Only collect pair data from specific keys
#                 if subkey in pair_keys:
#                     has_pair_data = True
#                     pair_ids = subdict.get('pair_ID', [])
#                     pair_bboxes = subdict.get('pair_bbox', [])
#                     if pair_ids:
#                         pair_ID_sets.append(set(pair_ids))
#                     if pair_bboxes:
#                         pair_bbox_sets.append(set(tuple(b) for b in pair_bboxes))
#
#         # Compute and assign target-level results
#         slot['final_ID'] = list(set.intersection(*target_ID_sets)) if target_ID_sets else []
#         slot['final_bbox'] = [list(b) for b in set.intersection(*target_bbox_sets)] if target_bbox_sets else []
#
#         # Conditionally assign pair-level results
#         if has_pair_data:
#             slot['final_ID_pair'] = list(set.intersection(*pair_ID_sets)) if pair_ID_sets else []
#             slot['final_bbox_pair'] = [list(b) for b in set.intersection(*pair_bbox_sets)] if pair_bbox_sets else []
#
#     return data













# Function to update the data_dict with the relevant attributes from selected_person
# Function to update the data_dict with the relevant attributes from selected_person



# def update_entities(selected_person, data_dict, target_Q):
#     # Step 1: Update the 'category'
#     data_dict['question']['entities']['category'] = target_Q
#
#     # Step 2: Loop through the keys in selected_person and add them to 'entities'
#     for key, value in selected_person.items():
#         if key == 'variation_slots':  # Handle variation_slots separately
#             for slot_key, slot_value in value.items():
#                 if 'variable' in slot_value:
#                     for var_key, var_value in slot_value['variable'].items():
#                         data_dict['question']['entities'][var_key] = var_value.get('variable', [])
#         else:
#             data_dict['question']['entities'][key] = value
#
#     return data_dict  # Return the modified data_dict
# def fill_all_slots(data):
#     pair_keys = {'hhi', 'hhg', 'hoi'}
#
#     for slot_key, slot in data.items():
#         target_pair_sets = []
#         pair_pair_sets = []
#         has_pair_data = False
#
#         for subkey, subdict in slot.items():
#             if isinstance(subdict, dict):
#                 ids = subdict.get('target_ID', [])
#                 bboxes = subdict.get('target_bbox', [])
#                 if ids and bboxes and len(ids) == len(bboxes):
#                     pairs = set(zip(ids, map(tuple, bboxes)))
#                     target_pair_sets.append(pairs)
#
#                 # handle pair_ID and pair_bbox for hhi/hhg/hoi
#                 if subkey in pair_keys:
#                     has_pair_data = True
#                     # pair_ids = subdict.get('pair_ID', [])
#                     # pair_bboxes = subdict.get('pair_bbox', [])
#                     # if pair_ids and pair_bboxes and len(pair_ids) == len(pair_bboxes):
#                     #     pair_pairs = set(zip(pair_ids, map(tuple, pair_bboxes)))
#                     #     pair_pair_sets.append(pair_pairs)
#                     pair_ids = subdict.get('pair_ID', [])
#                     pair_bboxes = subdict.get('pair_bbox', [])
#
#                     if pair_ids and pair_bboxes and len(pair_ids) == len(pair_bboxes):
#                         # Filter out any (id, bbox) where either is None
#                         clean_pairs = [
#                             (id_, tuple(bbox))
#                             for id_, bbox in zip(pair_ids, pair_bboxes)
#                             if id_ is not None and bbox is not None
#                         ]
#                         if clean_pairs:  # Only append if we have valid pairs
#                             pair_pairs = set(clean_pairs)
#                             pair_pair_sets.append(pair_pairs)
#
#         # Intersect aligned (ID, bbox) pairs
#         final_pairs = set.intersection(*target_pair_sets) if target_pair_sets else set()
#         slot['final_ID'] = [id_ for id_, _ in final_pairs]
#         slot['final_bbox'] = [list(bbox) for _, bbox in final_pairs]
#
#         # Same for pair-level data
#         if has_pair_data:
#             final_pair_pairs = set.intersection(*pair_pair_sets) if pair_pair_sets else set()
#             slot['final_ID_pair'] = [id_ for id_, _ in final_pair_pairs]
#             slot['final_bbox_pair'] = [list(bbox) for _, bbox in final_pair_pairs]
#
#     return data

def fill_all_slots(data):
    pair_keys = {'hhi', 'hhg', 'hoi'}

    def _clean_id_bbox(ids, bboxes):
        """
        Return a set of (id, bbox_tuple) after validating inputs.
        Skips any pair where id or bbox is None or bbox is non-iterable.
        """
        if not ids or not bboxes:
            return set()
        # Ensure both are iterable sequences
        if not isinstance(ids, (list, tuple)) or not isinstance(bboxes, (list, tuple)):
            return set()

        clean = []
        for id_, bbox in zip(ids, bboxes):
            if id_ is None or bbox is None:
                continue
            # Ensure bbox is iterable/convertible to tuple
            try:
                bbox_t = tuple(bbox)
            except TypeError:
                continue
            clean.append((id_, bbox_t))
        return set(clean)

    # If we write into `slot` while iterating, be safe by materializing items
    for slot_key, slot in list(data.items()):
        target_pair_sets = []
        pair_pair_sets = []
        has_pair_data = False

        for subkey, subdict in list(slot.items()):
            if isinstance(subdict, dict):
                ids = subdict.get('target_ID', None)
                bboxes = subdict.get('target_bbox', None)
                pairs = _clean_id_bbox(ids, bboxes)
                if pairs:
                    target_pair_sets.append(pairs)

                if subkey in pair_keys:
                    has_pair_data = True
                    pair_ids = subdict.get('pair_ID', None)
                    pair_bboxes = subdict.get('pair_bbox', None)
                    pair_pairs = _clean_id_bbox(pair_ids, pair_bboxes)
                    if pair_pairs:
                        pair_pair_sets.append(pair_pairs)

        # Intersect aligned (ID, bbox) pairs
        final_pairs = set.intersection(*target_pair_sets) if target_pair_sets else set()
        slot['final_ID'] = [id_ for id_, _ in final_pairs]
        slot['final_bbox'] = [list(bbox) for _, bbox in final_pairs]

        # Same for pair-level data
        if has_pair_data:
            final_pair_pairs = set.intersection(*pair_pair_sets) if pair_pair_sets else set()
            slot['final_ID_pair'] = [id_ for id_, _ in final_pair_pairs]
            slot['final_bbox_pair'] = [list(bbox) for _, bbox in final_pair_pairs]

    return data

def fill_all_slots_VQA(data):
    pair_keys = {'hhi', 'hhg', 'hoi'}

    for slot_key, slot in data.items():
        final_ids = set()
        final_answers = set()

        for subkey, subdict in slot.items():
            if isinstance(subdict, dict):
                # Regular ID/answer (e.g., race, action)
                ids = subdict.get('target_ID', [])
                answers = subdict.get('target_answer', [])

                final_ids.update(ids)
                final_answers.update(answers)

                # Skip pair logic unless it's relevant
                if subkey in pair_keys:
                    pair_ids = subdict.get('pair_ID', [])
                    pair_answers = subdict.get('pair_answer', [])
                    final_ids.update(pair_ids)
                    for ans in pair_answers:
                        if isinstance(ans, list):
                            final_answers.update(ans)
                        else:
                            final_answers.add(ans)

        # Assign the union results back
        slot['final_ID'] = list(final_ids)
        slot['final_answer'] = list(final_answers)

    return data


from typing import Dict, Any

def update_entities(
    selected_person: Dict[str, Any],
    data_dict: Dict[str, Any],
    target_category: str,  # Still passed, but not used anymore
) -> Dict[str, Any]:
    """
    Copy selected_person into entities while:
    - NOT including 'category'
    - Removing 'start' and 'end' inside variation_slots > slot > variable > hhg/hhi/hoi
    """
    entities = data_dict.setdefault("question", {}).setdefault("entities", {})

    # Copy selected_person into entities
    for key, value in selected_person.items():
        if key == "variation_slots" and isinstance(value, dict):
            cleaned_slots = {}
            for slot_key, slot_val in value.items():
                cleaned_slot = dict(slot_val)  # shallow copy
                if "variable" in slot_val and isinstance(slot_val["variable"], dict):
                    cleaned_variables = {}
                    for var_key, var_val in slot_val["variable"].items():
                        if var_key in ("hhg", "hhi", "hoi") and isinstance(var_val, dict):
                            # Remove 'start' and 'end' from hhg/hhi/hoi
                            cleaned_var = {
                                k: v for k, v in var_val.items()
                                if k not in ("start", "end")
                            }
                            cleaned_variables[var_key] = cleaned_var
                        else:
                            cleaned_variables[var_key] = var_val
                    cleaned_slot["variable"] = cleaned_variables
                cleaned_slots[slot_key] = cleaned_slot
            entities["variation_slots"] = cleaned_slots
        else:
            entities[key] = value

    return data_dict

def create_data_dict(Set='train',
                     task_type='VG',
                     S='2',
                     T=2,
                     target_Q='human',
                     modality_type='video',
                     seq_name='sequence_001'):
    """
    Create a DataDictionary-style dictionary with structured fields.

    Returns:
        dict: Structured in the style of a custom DataDictionary class.
    """
    spatial_level = int(S)
    difficulty = spatial_level + T
    question_level = "U(S)P(T)"

    data_dict = {
        "task_type": task_type,
        "type": target_Q,
        "difficulty_level": difficulty,
        "question_level": question_level,
        "data": {
            "seq": seq_name,
            "modality_type": modality_type,
            "specs": {
                "fps": 15
            }
        },
        "question": {
            "question": {},  # you can populate this later
            "spatial_level": spatial_level,
            "temporal_level": T,
            "entities": {},
            "timestamps": {}
        },
        "labels": {}
    }

    return data_dict


import numpy as np

def evenly_sample_items(all_numbers, N):
    """
    Evenly sample N items from a list of sorted elements (e.g., image filenames).

    Args:
        all_numbers (list): List of items to sample from (e.g., ['000000.jpg', ..., '000050.jpg']).
        N (int): Number of items to sample.

    Returns:
        list: Evenly sampled items from the input list.
    """
    all_numbers = sorted(all_numbers)  # ensure sorted
    indices = np.linspace(0, len(all_numbers) - 1, N).astype(int)
    sampled_numbers = [all_numbers[i] for i in indices]
    return sampled_numbers

import os

def ensure_folder_exists(folder_name: str):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created.")
    else:
        print(f"Folder '{folder_name}' already exists.")

def read_pose(seq_name):
    path_pose = f'/Users/sjah0003/PycharmProjects/pythonProject_visulizastion/labels/labels_2d_pose_stitched_coco/{seq_name}.json'
    with open(path_pose, "r") as file:
        data = json.load(file)
    return data['annotations']
def Face_visibility(person,data_annotation_pose,image):
    face_visibility = 'invisible'

    person_id = int(person.get('label_id').split(':')[1])
    image_id = int(image.split('.')[0])
    # data_annotation_pose = read_pose(seq_name)
    for i in data_annotation_pose['annotations']:
        if i.get('track_id') == person_id and (i.get('image_id')-2) == image_id:
           key_points_person =  i.get('keypoints')
           # if key_points_person[5] == 2 and key_points_person[8] == 2:
           #    face_visibility = 'visible'
           if key_points_person[5] == 0 and key_points_person[8] == 0:
              face_visibility = 'invisible'
           elif key_points_person[5] != 0 and key_points_person[8] != 0:
               face_visibility = 'visible'
           break
    return face_visibility




