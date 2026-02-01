import itertools
from global_functions import *
import copy
def data_dictionary_to_dict(data_dict):
    """
    Convert a DataDictionary object to a dictionary.

    :param data_dict: DataDictionary object
    :return: Dictionary representation of the DataDictionary object
    """
    return {
        "task_type": data_dict.task_type,
        "type": data_dict.type,
        "difficulty_level": data_dict.difficulty_level,
        "question_level": data_dict.question_level,
        "data": data_dict.data,
        "question": data_dict.question,
        "labels": data_dict.labels
    }
def generate_combinations(S, target_Q, T=1):
    """
    Generates attribute combinations based on scope level (S), target entity (target_Q), and combination level (T).

    Parameters:
    S (str): Scope level ('1', '2', or '3').
    target_Q (str): Target entity ('human', 'object', or 'human&object').
    T (int, optional): Combination level, default is 1.

    Returns:
    list: A list of valid attribute combinations.
    """
    attribute_sets = {
        '1': {
            'human': {
                'attributes': {
                    'age': 'age_R',
                    'race': 'race_R',
                    'gender': 'gender_R',
                    'action_labels': 'action_labels_R',
                    # 'h_dis_robot': 'dis_R',
                    # 'SR_robot_ref': 'SR_robot_ref'
                },
                'required': {'action_labels', 'h_dis_robot', 'SR_robot_ref'}
            },
            'object': {
                'attributes': {
                    'category': 'category_R',
                    'obj_dis_robot': 'D_R',
                    'SR_robot_ref': 'SR_robot_ref'
                },
                'required': {'obj_dis_robot', 'SR_robot_ref'}
            }
        },
        '2': {
            'human': {
                'attributes': {
                    'age': 'age_R',
                    'gender': 'gender_R',
                    'hhi': 'hhi_R',
                    'hhG': 'hhG_R'
                },
                'required': {'hhi', 'hhG'}
            },
            'human&object': {
                'attributes': {
                    'age': 'age_R',
                    'race': 'race_R',
                    'gender': 'gender_R',
                    'hoi': 'hoi_R',
                    'hoG': 'hoG_R'
                },
                'required': {'hoi', 'hoG'}
            }
        },
        '3': {
            'human': {
                'attributes': {
                    'age': 'age_R',
                    'race': 'race_R',
                    'gender': 'gender_R',
                    'hhG': 'hhG_R',
                    'hhi': 'hhi_R',
                    'group_size': 'group_size'
                },
                'required': {'hhi', 'hhG'}
            },
            'human&object': {
                'attributes': {
                    'age': 'age_R',
                    'race': 'race_R',
                    'gender': 'gender_R',
                    'hoi': 'hoi_R',
                    'hoG': 'hoG_R',
                    'group_size': 'group_size'
                },
                'required': {'hoi', 'hoG'}
            }
        }
    }

    if S not in attribute_sets or target_Q not in attribute_sets[S]:
        return []

    attributes = attribute_sets[S][target_Q]['attributes']
    required_attributes = attribute_sets[S][target_Q]['required']
    attribute_keys = list(attributes.keys())
    all_combinations = []

    for r in range(1, len(attribute_keys) + 1):
        combinations = list(itertools.combinations(attribute_keys, r))
        if S == '3':
            # Ensure all required attributes are present
            combinations = [list(comb) for comb in combinations if required_attributes.issubset(comb)]
        elif S == '2':
            # Ensure at least one required attribute is present, but not all
            combinations = [list(comb) for comb in combinations if
                            required_attributes.intersection(comb) and not required_attributes.issubset(comb)]
        elif T > 1:
            # Ensure at least one required attribute is present for scope 1
            combinations = [list(comb) for comb in combinations if required_attributes.intersection(comb)]
        else:
            combinations = [list(comb) for comb in combinations]

        all_combinations.extend(combinations)

    return all_combinations
def VG_task(SpatialTemporalSceneGraph, S, T, target_Q,modality_type, data_dict, data_points, Incremental):
    all_combinations = get_all_combinations(S, T, target_Q)
    # Randomly pick one combination
    attributes_to_use = random.choice(all_combinations)
    print('attributes_to_use',attributes_to_use)
    # c=0
    # for attributes_to_use in all_combinations:
    for i in range(1):
        # attributes_to_use = ["hhg"]
        # data_dict = create_data_dict(Set=set,
        #                              task_type=task_type,
        #                              S=S,
        #                              T=T,
        #                              target_Q=target_Q,
        #                              modality_type=modality_type,
        #                              seq_name=seq)
        # raw_Data_slots= SpatialTemporalSceneGraph.extract_entity_time_slots(attributes_to_use=attributes_to_use, entity_type="human")
        # Data_all = extract_different_value_combinations_all_keys(raw_Data_slots, T=T)   # doros ta enja
        #
        # filtered_Data_all = {
        #     pid: info
        #     for pid, info in Data_all.items()
        #     if all(attr in info for attr in attributes_to_use)
        # }
        # # all_possible_persons = {}
        # # for person, value in Data_all.items():
        # #     data = cal_over_laps(value, T, S)
        # #     if person not in all_possible_persons and data:
        # #         all_possible_persons[person] = data
        # #
        # # # all possible ones
        # # all_possible_persons
        #
        # Data_all_combinations = {}
        # mandatory_demo = ('age', 'race', 'gender')
        #
        # for person, value in filtered_Data_all.items():
        #     # 1. find which demo keys are actually in this record
        #     present_demos = [k for k in mandatory_demo if k in value]
        #
        #     # 2. pull out those keys
        #     demo_info = {k: value[k] for k in present_demos}
        #
        #     # 3. if any of those present demo values is None, skip entirely
        #     if any(v is None for v in demo_info.values()):
        #         continue
        #
        #     # 4. strip all three demo keys (so overlap-logic never sees them)
        #     value_no_demo = {k: v for k, v in value.items() if k not in mandatory_demo}
        #
        #     # 5. compute your overlaps
        #     overlaps = cal_over_laps(value_no_demo, T, S)
        #
        #     # 6. only store people who have non‐empty overlaps
        #     if overlaps:
        #         Data_all_combinations[person] = {
        #             'combinations': overlaps,
        #             **demo_info  # reattach whichever demo keys we pulled out
        #         }
        #
        # if not Data_all_combinations:
        #    continue
        # selected_person = pick_random_person(Data_all_combinations)
        # attributes_to_use = ["distance"]  # Can be any subset of ['age', 'race', 'gender'] SR_robot_ref
        # attributes_to_use=['hhg']
        # attributes_to_use = ['age', 'race', 'gender', 'hhi','hhg']
        # c+=1
        # if c==8:
        #     # break
        #     print('g')
        # attributes_to_use = ['age','race','gender','hhi','hhg']
        # attributes_to_use = ['hhg']
        # print('attributes_to_use',attributes_to_use)
        # attributes_to_use = random.choice(all_combinations)
        # if S == "1":
            # attributes_to_use = ['gender','age','race','distance', 'SR_Robot_Ref']   #s=1
            # attributes_to_use = ['age']  # s=1
        # elif S == "2":
        #     attributes_to_use = ['gender','age','race','hhg']  #s=2
        # elif S == "3":
        #     attributes_to_use = ['gender','age','race', 'hhi','hhg']  #s=3
        # print('attributes_to_use',attributes_to_use)

        data_dict_copy = copy.deepcopy(data_dict)

        keys_to_check = ['age', 'race', 'gender']

        if any(key in attributes_to_use for key in keys_to_check):
            # print("All keys are present.")
            attributes_to_use.append('face_visibility')
        # 2. Extract raw entity time slots
        raw_Data_slots = SpatialTemporalSceneGraph.extract_entity_time_slots(
            attributes_to_use=attributes_to_use,
            entity_type=target_Q
        )

        # 3. Extract combinations of attribute values for all keys
        Data_all = extract_different_value_combinations_all_keys(raw_Data_slots, T=T)

        # 4. Filter people who have all requested attributes
        filtered_Data_all = {
            pid: info
            for pid, info in Data_all.items()
            if all(attr in info and info[attr] is not None for attr in attributes_to_use)
        }

        # # 5. Initialize the result
        # Data_all_combinations = {}
        #
        # # define which keys you consider "demographic"
        # DEMOGRAPHIC_KEYS = {"age", "race", "gender"}
        #
        # # 6. For each person, extract attributes and overlaps
        # for person, value in filtered_Data_all.items():
        #     # a. Of the attributes_to_use list, pick only the demographics
        #     demo_keys = [k for k in attributes_to_use if k in DEMOGRAPHIC_KEYS]
        #
        #     # b. Build your demo_info from those keys
        #     demo_info = {k: value[k] for k in demo_keys if k in value}
        #
        #     # c. Strip out *only* demographics, leaving all other fields
        #     value_no_demo = {
        #         k: v
        #         for k, v in value.items()
        #         if k not in DEMOGRAPHIC_KEYS
        #     }
        #
        #     # d. If there's nothing left to overlap on, still include this person
        #     if not value_no_demo:
        #         Data_all_combinations[person] = {
        #             "combinations": [],  # no overlap attrs
        #             **demo_info
        #         }
        #     else:
        #         # otherwise compute overlaps as before
        #         overlaps = cal_over_laps(value_no_demo, T, S)
        #         if overlaps:
        #             Data_all_combinations[person] = {
        #                 "combinations": overlaps,
        #                 **demo_info
        #             }
        # === top‐level assembly ===

        # filtered_Data_all: dict mapping person → dict of all their attributes
        # attributes_to_use: list of keys to include (e.g. ["age","race","action"])
        # T, S: parameters passed into cal_over_laps

        # Data_all_combinations = {}
        # DEMOGRAPHIC_KEYS = {"age", "race", "gender"}
        #
        # for person, value in filtered_Data_all.items():
        #     # a) pick out only the demographic fields from attributes_to_use
        #     demo_keys = [k for k in attributes_to_use if k in DEMOGRAPHIC_KEYS]
        #     demo_info = {k: value[k] for k in demo_keys if k in value}
        #
        #     # b) pick only non‐demographic fields from attributes_to_use
        #     overlap_keys = [k for k in attributes_to_use if k not in DEMOGRAPHIC_KEYS]
        #     value_no_demo = {k: value[k] for k in overlap_keys if k in value}
        #
        #     # c) compute overlaps (or get empty list if none)
        #     if value_no_demo:
        #         overlaps = cal_over_laps(value_no_demo, T, S) or []
        #     else:
        #         overlaps = []
        #
        #     # d) always include this person, with consistent structure
        #     Data_all_combinations[person] = {
        #         "combinations": overlaps,
        #         **demo_info
        #     }

        # Data_all_combinations = {}
        # DEMOGRAPHIC_KEYS = {"age", "race", "gender"}
        #
        # for person, value in filtered_Data_all.items():
        #     # a) pick out only the demographic fields from attributes_to_use
        #     demo_keys = [k for k in attributes_to_use if k in DEMOGRAPHIC_KEYS]
        #     demo_info = {k: value[k] for k in demo_keys if k in value}
        #
        #     # b) pick only non‐demographic fields from attributes_to_use
        #     overlap_keys = [k for k in attributes_to_use if k not in DEMOGRAPHIC_KEYS]
        #     value_no_demo = {k: value[k] for k in overlap_keys if k in value}
        #
        #     # 🔧 c) always attempt overlap calculation, even if value_no_demo is empty
        #     overlaps = cal_over_laps(value_no_demo, T, S) if value_no_demo else [{} for _ in range(1)]
        #
        #     # d) always include this person, with consistent structure
        #     Data_all_combinations[person] = {
        #         "combinations": overlaps,
        #         **demo_info
        #     }
        Data_all_combinations = {}
        DEMOGRAPHIC_KEYS = {"age", "race", "gender"}

        for person, value in filtered_Data_all.items():
            # a) Demographics
            demo_keys = [k for k in attributes_to_use if k in DEMOGRAPHIC_KEYS]
            demo_info = {k: value[k] for k in demo_keys if k in value}

            # b) Non-demographics
            overlap_keys = [k for k in attributes_to_use if k not in DEMOGRAPHIC_KEYS]
            value_no_demo = {k: value[k] for k in overlap_keys if k in value}

            # c) Compute overlaps or create dummy using inferred start/end
            if value_no_demo:
                overlaps = cal_over_laps(value_no_demo, T, S) or []
            else:
                # Try to infer start/end from demographic fields
                inferred_start, inferred_end = 0, 0  # fallback default
                for k in demo_keys:
                    if k in value and value[k]:
                        interval = value[k][0][0]  # unpack first interval tuple
                        inferred_start = interval.get("start", inferred_start)
                        inferred_end = interval.get("end", inferred_end)
                        break  # use the first valid one we find

                # dummy_type = random.choice(["specific", "vague"])
                dummy_type = random.choice(["specific"])
                dummy_combination = {
                    f"slot{t}": {
                        "start": inferred_start,
                        "end": inferred_end,
                        "variable": {},
                        "type": dummy_type
                    } for t in range(1, T + 1)
                }
                overlaps = [dummy_combination]

            # d) Final output
            Data_all_combinations[person] = {
                "combinations": overlaps,
                **demo_info
            }

        if not Data_all_combinations:
           continue

        if modality_type == "image":
           if any(key in attributes_to_use for key in keys_to_check):
               visibility='invisible'
               while visibility =='invisible':
                     selected_person = pick_random_person(Data_all_combinations, attributes_to_use=attributes_to_use)
                     selected_person = specialize_for_image(selected_person)
                     # print("selected_person",selected_person)
                     visibility = selected_person['variation_slots']['slot1']['variable'].get('face_visibility')['variable'][0]
           else:
               selected_person = pick_random_person(Data_all_combinations, attributes_to_use=attributes_to_use)
               selected_person = specialize_for_image(selected_person)

        else: # Modality is video
            selected_person = pick_random_person(Data_all_combinations, attributes_to_use=attributes_to_use)

        if 'face_visibility' in attributes_to_use:
            attributes_to_use.remove('face_visibility')
        attributes_to_use_new = attributes_to_use.copy()
        if target_Q in ['human&object','human']:
           attributes_to_use_new.append('human')
        else:
            attributes_to_use_new.append('object')
        # attributes_to_use_new.append('node_name')
        final_structures = find_attributes_in_graph(selected_person, attributes_to_use_new, SpatialTemporalSceneGraph)

        if final_structures is None:
            continue
        # print(final_structures)
        bbox_number = get_final_bbox_length(final_structures)
        Question = Refinement_attribute_person_T(selected_person, bbox_number)
        data_dict_copy['question']['question'] = Question

        data_dict_copy['labels'] = final_structures
        # data_dict['question']['entities']['category'] = target_Q
        data_dict_copy = update_entities(selected_person, data_dict_copy, target_Q)

        # results = split_and_refine(data_dict_copy['question']['entities'], bbox_number=1)
        # results = build_single_attribute_descriptions(data_dict_copy['question']['entities'])
        # results = build_ordered_attribute_descriptions(data_dict_copy['question']['entities'])
        # results = build_ordered_attribute_descriptions(data_dict_copy['question']['entities'], attributes_to_use)
        results = build_ordered_attribute_descriptions(data_dict_copy['question']['entities'], attributes_to_use, incremental=Incremental)
        results["human"] = "Find all persons"
        data_dict_copy['question']['sub_questions'] = results

        updated_final_structures = update_final_structures_by_results(final_structures, results)
        data_dict_copy['labels'] = updated_final_structures


        if 'timestamps' not in data_dict_copy['question']:
            data_dict_copy['question']['timestamps'] = {}

        # for slot, value in Slots.items():
        Slots = selected_person.get("variation_slots")
        for slot, value in Slots.items():
            file_id_list = [f"{i:06}.jpg" for i in range(value['start'], value['end'] + 1)]
            data_dict_copy['question']['timestamps'][slot] = {
                't1': float(value['start']),
                't2': float(value['end']),
                'image_ids': file_id_list,
                'type': value['type']
            }
        data_points.append(data_dict_copy)
        print('Question',Question)
        print('entity', data_dict_copy['question']['entities'])
        print('Add one sample')
        # break
    return data_points