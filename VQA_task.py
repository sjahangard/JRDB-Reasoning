import itertools
from global_functions import *
import copy
def VQA_task(SpatialTemporalSceneGraph, S, T, target_Q,modality_type, data_dict, data_points, Incremental):

    all_combinations = get_all_combinations(S, T, target_Q)
    # Filter out sublists with length 1
    all_combinations = [item for item in all_combinations if len(item) > 1]
    for i in range(1):
    # for attributes_to_use in all_combinations:
        # attributes_to_use = ['race', 'action', 'distance', 'SR_Robot_Ref']
        # attributes_to_use = random.choice(all_combinations)
        # # data_dict_copy = copy.deepcopy(data_dict)
        # output = generate_ask_provide_combinations(attributes_to_use, T)
        # picked_attributes = random.choice(output)
        # # picked_attributes = (['race', 'action'], ['distance', 'SR_Robot_Ref'])
        # ask, provide = picked_attributes

        ### S=1 ######----------
        # ask = ['gender', 'age', 'action']
        # provide = ['distance']

        # ### S=2 ######----------
        # ask = ['gender', 'age', 'action','hhi']
        # provide = ['distance']

        ## S=3 ######----------
        ask = ['gender', 'age', 'action', 'hhi','hhg']
        provide = ['distance']

        data_dict_copy = copy.deepcopy(data_dict)

        keys_to_check = ['age', 'race', 'gender']

        if any(key in provide for key in keys_to_check):
            # print("All keys are present.")
            provide.append('face_visibility')
        # 2. Extract raw entity time slots
        raw_Data_slots = SpatialTemporalSceneGraph.extract_entity_time_slots(
            attributes_to_use=provide,
            entity_type=target_Q
        )

        # 3. Extract combinations of attribute values for all keys
        Data_all = extract_different_value_combinations_all_keys(raw_Data_slots, T=T)

        # 4. Filter people who have all requested attributes
        filtered_Data_all = {
            pid: info
            for pid, info in Data_all.items()
            if all(attr in info and info[attr] is not None for attr in provide)
        }

        Data_all_combinations = {}
        DEMOGRAPHIC_KEYS = {"age", "race", "gender"}

        for person, value in filtered_Data_all.items():
            # a) Demographics
            demo_keys = [k for k in provide if k in DEMOGRAPHIC_KEYS]
            demo_info = {k: value[k] for k in demo_keys if k in value}

            # b) Non-demographics
            overlap_keys = [k for k in provide if k not in DEMOGRAPHIC_KEYS]
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
           if any(key in provide for key in keys_to_check):
               visibility='invisible'
               while visibility =='invisible':
                     selected_person = pick_random_person(Data_all_combinations, attributes_to_use=provide)
                     selected_person = specialize_for_image(selected_person)
                     print(selected_person)
                     visibility = selected_person['variation_slots']['slot1']['variable'].get('face_visibility')['variable'][0]
           else:
               selected_person = pick_random_person(Data_all_combinations, attributes_to_use=provide)
               selected_person = specialize_for_image(selected_person)

        else:
            selected_person = pick_random_person(Data_all_combinations, attributes_to_use=provide)

        if 'face_visibility' in provide:
            provide.remove('face_visibility')
        provide_new = provide.copy()
        if target_Q in ['human&object','human']:
           provide_new.append('human')
        # attributes_to_use_new.append('node_name')
        final_structures = find_attributes_in_graph_VQA(selected_person, ask, provide_new, SpatialTemporalSceneGraph)

        if final_structures is None:
            continue
        # print(final_structures)
        bbox_number = get_final_bbox_length(final_structures)
        # Question = Refinement_attribute_person_T(selected_person, bbox_number)
        Question = Refinement_attribute_person_T_VQA_Wh(selected_person, ask, provide, T, bbox_number)
        data_dict_copy['question']['question'] = Question

        data_dict_copy['labels'] = final_structures
        # data_dict['question']['entities']['category'] = target_Q
        data_dict_copy = update_entities(selected_person, data_dict_copy, target_Q)

        results = build_ordered_attribute_descriptions_VQA(data_dict_copy['question']['entities'],ask, provide, T, incremental=Incremental)
        # results["human"] = "Find all persons"
        data_dict_copy['question']['sub_questions'] = results

        updated_final_structures = update_final_structures_by_results_VQA(final_structures, results)
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


    return data_points