# import networkx as nx
# import matplotlib.pyplot as plt
# import numpy as np
#
# class SpatialTemporalSceneGraph:
#     def __init__(self):
#         self.graph = nx.MultiDiGraph()  # allows multiple edges between nodes
#         self.frame_data = {}            # Stores data per frame
#
#     # def add_human(self, frame_id, age, race, gender, action, social_group_id, group_size, bbox, human_id_in_frame,
#     #               occlusion):
#     #     if frame_id not in self.frame_data:
#     #         self.frame_data[frame_id] = {}
#     #     key = f"h_{human_id_in_frame}"
#     #     self.frame_data[frame_id][key] = {
#     #         'type': 'human', 'age': age, 'race': race, 'gender': gender,
#     #         'action': action, 'social_group_id': social_group_id,
#     #         'group_size': group_size, 'bbox': bbox, 'occlusion': occlusion
#     #     }
#     #     # node_name = f"{frame_id}_{key}"
#     #     node_name = f"h_{key}"
#     #     self.graph.add_node(node_name,
#     #                         node_type='human', age=age, race=race,
#     #                         gender=gender, action=action,
#     #                         social_group_id=social_group_id,
#     #                         group_size=group_size, bbox=bbox,
#     #                         ID=human_id_in_frame, occlusion=occlusion)
#     def add_human(self, frame_id, age, race, gender, action, social_group_id,
#                   group_size, bbox, human_id_in_frame, occlusion,
#                   SR_Robot_Ref=None, distance=None):
#         """
#         Add a human node with optional spatial-reference to robot and distance.
#
#         Args:
#             frame_id (str): Frame identifier.
#             age (str): Age category.
#             race (str): Race category.
#             gender (str): Gender category.
#             action (str): Current action.
#             social_group_id (int): Social group identifier.
#             group_size (int): Size of the group.
#             bbox (tuple): Bounding box coordinates (x1, y1, x2, y2).
#             human_id_in_frame (int): Unique human ID within the frame.
#             occlusion (float): Occlusion ratio.
#             SR_Robot_Ref (str, optional): Spatial reference ID to robot.
#             distance (float, optional): Distance to the reference robot.
#         """
#         # Initialize frame storage
#         if frame_id not in self.frame_data:
#             self.frame_data[frame_id] = {}
#         key = f"h_{human_id_in_frame}"
#         # Store in frame_data
#         self.frame_data[frame_id][key] = {
#             'type': 'human',
#             'age': age,
#             'race': race,
#             'gender': gender,
#             'action': action,
#             'social_group_id': social_group_id,
#             'group_size': group_size,
#             'bbox': bbox,
#             'occlusion': occlusion,
#             'SR_Robot_Ref': SR_Robot_Ref,
#             'distance': distance
#         }
#         # Create node in the graph
#         node_name = f"h_{frame_id}_{human_id_in_frame}"
#         self.graph.add_node(
#             node_name,
#             node_type='human',
#             age=age,
#             race=race,
#             gender=gender,
#             action=action,
#             social_group_id=social_group_id,
#             group_size=group_size,
#             bbox=bbox,
#             occlusion=occlusion,
#             SR_Robot_Ref=SR_Robot_Ref,
#             distance=distance,
#             ID=human_id_in_frame
#         )
#     def add_object(self, frame_id, obj_name, tracking_id, category_id, bbox=None):
#         if frame_id not in self.frame_data:
#             self.frame_data[frame_id] = {}
#         # key = f"o_{tracking_id}_{category_id}"
#         key = f"o_{tracking_id}_{category_id}"
#         self.frame_data[frame_id][key] = {
#             'type': 'object', 'obj_name': obj_name,
#             'tracking_id': tracking_id, 'category_id': category_id,
#             'bbox': bbox
#         }
#         node_name = f"{frame_id}_{key}"
#         self.graph.add_node(node_name,
#                             node_type='object', obj_name=obj_name,
#                             tracking_id=tracking_id, category_id=category_id,
#                             bbox=bbox, ID=f"{tracking_id}_{category_id}")
#
#     def add_physical_relationship(self, frame_id, node_key_1, node_key_2, relation_type):
#         self.graph.add_edge(f"{frame_id}_{node_key_1}",
#                             f"{frame_id}_{node_key_2}",
#                             relation_type=relation_type,
#                             edge_type='physical')
#
#     def add_geometric_relationship(self, frame_id, node_key_1, node_key_2, relation_type):
#         self.graph.add_edge(f"{frame_id}_{node_key_1}",
#                             f"{frame_id}_{node_key_2}",
#                             relation_type=relation_type,
#                             edge_type='geometric')
#
#     def get_object_at_frame(self, frame_id):
#         return self.frame_data.get(frame_id, {})
#
#     def get_relationships(self):
#         return self.graph.edges(data=True, keys=True)
#
#     def get_physical_relationships(self):
#         return [(u, v, k, d) for u, v, k, d in self.graph.edges(data=True, keys=True)
#                 if d.get('edge_type') == 'physical']
#
#     def get_geometric_relationships(self):
#         return [(u, v, k, d) for u, v, k, d in self.graph.edges(data=True, keys=True)
#                 if d.get('edge_type') == 'geometric']
#
#     # def extract_entity_time_slots(self, attributes_to_use=None, entity_type="human"):
#     #     """
#     #     Group frame-by-frame values (including HHI/HOI interactions) into contiguous time slots per entity.
#     #     Supports node attributes (e.g., action) and edge-based interactions:
#     #       - "HHI": human-human interactions (physical edges between humans), returns list of (other_id, relation)
#     #       - "HOI": human-object interactions (physical edges human→object), returns list of (object_id, relation)
#     #     """
#     #     if attributes_to_use is None:
#     #         attributes_to_use = ["action"]
#     #
#     #     # Sort frames by numeric part of filename
#     #     frame_ids = sorted(
#     #         self.frame_data.keys(),
#     #         key=lambda x: int(x.split('.')[0])
#     #     )
#     #     entity_data = {}
#     #
#     #     # Collect per-frame history
#     #     for frame_id in frame_ids:
#     #         frame_num = int(frame_id.split('.')[0])
#     #         for node_key, data in self.frame_data[frame_id].items():
#     #             if data.get('type') != entity_type:
#     #                 continue
#     #             entity_id = str(data.get('ID', node_key))
#     #             if entity_id not in entity_data:
#     #                 entity_data[entity_id] = {attr: [] for attr in attributes_to_use}
#     #                 entity_data[entity_id]['_history'] = {attr: [] for attr in attributes_to_use}
#     #
#     #             # for attr in attributes_to_use:
#     #             #     src = f"{frame_id}_{node_key}"
#     #             #     if attr == "hhi":
#     #             #         interactions = []
#     #             #         for _, tgt, _, edge_data in self.graph.out_edges(src, data=True, keys=True):
#     #             #             if edge_data.get('edge_type') == 'physical' and \
#     #             #                self.graph.nodes[tgt].get('node_type') == 'human':
#     #             #                 other_id = self.graph.nodes[tgt]['ID']
#     #             #                 interactions.append((other_id, edge_data.get('relation_type')))
#     #             #         val = tuple(sorted(interactions, key=lambda x: x[0])) if interactions else None
#     #             #     elif attr == "hoi":
#     #             #         interactions = []
#     #             #         for _, tgt, _, edge_data in self.graph.out_edges(src, data=True, keys=True):
#     #             #             if edge_data.get('edge_type') == 'physical' and \
#     #             #                self.graph.nodes[tgt].get('node_type') == 'object':
#     #             #                 obj_id = self.graph.nodes[tgt]['ID']
#     #             #                 interactions.append((obj_id, edge_data.get('relation_type')))
#     #             #         val = tuple(sorted(interactions, key=lambda x: x[0])) if interactions else None
#     #             #     else:
#     #             #         val = data.get(attr)
#     #             #     entity_data[entity_id]['_history'][attr].append((frame_num, val))
#     #             for attr in attributes_to_use:
#     #                 if node_key.startswith('h_'):
#     #                     human_id = node_key.split('_')[1]
#     #                     src = f"h_{frame_id}_{human_id}"
#     #                 else:
#     #                     continue  # skip if not human
#     #
#     #                 if attr == "hhi":
#     #                     interactions = []
#     #                     for _, tgt, _, edge_data in self.graph.out_edges(src, data=True, keys=True):
#     #                         if edge_data.get('edge_type') == 'physical' and \
#     #                                 self.graph.nodes[tgt].get('node_type') == 'human':
#     #                             other_id = self.graph.nodes[tgt]['node_name']
#     #                             interactions.append((other_id, edge_data.get('relation_type')))
#     #                     val = tuple(sorted(interactions, key=lambda x: x[0])) if interactions else None
#     #                 elif attr == "hoi":
#     #                     interactions = []
#     #                     for _, tgt, _, edge_data in self.graph.out_edges(src, data=True, keys=True):
#     #                         if edge_data.get('edge_type') == 'physical' and \
#     #                                 self.graph.nodes[tgt].get('node_type') == 'object':
#     #                             obj_id = self.graph.nodes[tgt]['ID']
#     #                             interactions.append((obj_id, edge_data.get('relation_type')))
#     #                     val = tuple(sorted(interactions, key=lambda x: x[0])) if interactions else None
#     #                 else:
#     #                     val = data.get(attr)
#     #
#     #                 entity_data[entity_id]['_history'][attr].append((frame_num, val))
#     #
#     #     # Helper to merge contiguous values
#     #     def group_intervals(history):
#     #         intervals = []
#     #         if not history:
#     #             return intervals
#     #         prev_frame = history[0][0]
#     #         start, current_val = history[0]
#     #         for frame, val in history[1:]:
#     #             if val != current_val:
#     #                 intervals.append({'start': start, 'end': prev_frame, 'value': current_val})
#     #                 start, current_val = frame, val
#     #             prev_frame = frame
#     #         intervals.append({'start': start, 'end': prev_frame, 'value': current_val})
#     #         return intervals
#     #
#     #     # Build and return time slots
#     #     for ent, data in entity_data.items():
#     #         for attr in attributes_to_use:
#     #             data[attr] = group_intervals(data['_history'][attr])
#     #         data.pop('_history', None)
#     #
#     #     return entity_data
#     # def extract_entity_time_slots(self, attributes_to_use=None, entity_type="human"):
#     #     """
#     #     Group frame-by-frame values (including HHI/HOI interactions) into contiguous time slots per entity.
#     #     Supports node attributes (e.g., action) and edge-based interactions:
#     #       - "hhi": human-human interactions (physical edges between humans), returns list of (other_id, relation)
#     #       - "hoi":  human-object interactions (physical edges human→object), returns list of (object_id, relation)
#     #     """
#     #     if attributes_to_use is None:
#     #         attributes_to_use = ["action"]
#     #
#     #     # Sort frames by numeric part of filename
#     #     frame_ids = sorted(
#     #         self.frame_data.keys(),
#     #         key=lambda x: int(x.split('.')[0])
#     #     )
#     #     entity_data = {}
#     #
#     #     # Collect per-frame history
#     #     for frame_id in frame_ids:
#     #         frame_num = int(frame_id.split('.')[0])
#     #         for node_key, data in self.frame_data[frame_id].items():
#     #             if data.get('type') != entity_type:
#     #                 continue
#     #
#     #             entity_id = str(data.get('ID', node_key))
#     #             if entity_id not in entity_data:
#     #                 entity_data[entity_id] = {attr: [] for attr in attributes_to_use}
#     #                 entity_data[entity_id]['_history'] = {attr: [] for attr in attributes_to_use}
#     #
#     #             # Build the graph-node name just as in add_human():
#     #             # node_key is like "h_<human_id>"
#     #             if node_key.startswith('h_'):
#     #                 human_id = node_key.split('_')[1]
#     #                 src = f"h_{frame_id}_{human_id}"
#     #             else:
#     #                 # skip non-human if entity_type=="human"
#     #                 continue
#     #
#     #             for attr in attributes_to_use:
#     #                 if attr == "hhi":
#     #                     interactions = []
#     #                     for _, tgt, _, edge_data in self.graph.out_edges(src, data=True, keys=True):
#     #                         if (edge_data.get('edge_type') == 'physical'
#     #                                 and self.graph.nodes[tgt].get('node_type') == 'human'):
#     #                             other_id = self.graph.nodes[tgt]['ID']
#     #                             rel = edge_data.get('relation_type')
#     #                             interactions.append((other_id, rel))
#     #                     val = tuple(sorted(interactions, key=lambda x: x[0])) if interactions else None
#     #
#     #                 elif attr == "hoi":
#     #                     interactions = []
#     #                     for _, tgt, _, edge_data in self.graph.out_edges(src, data=True, keys=True):
#     #                         if (edge_data.get('edge_type') == 'physical'
#     #                                 and self.graph.nodes[tgt].get('node_type') == 'object'):
#     #                             obj_id = self.graph.nodes[tgt]['ID']
#     #                             rel = edge_data.get('relation_type')
#     #                             interactions.append((obj_id, rel))
#     #                     val = tuple(sorted(interactions, key=lambda x: x[0])) if interactions else None
#     #
#     #                 else:
#     #                     val = data.get(attr)
#     #
#     #                 entity_data[entity_id]['_history'][attr].append((frame_num, val))
#     #
#     #     # Helper to merge contiguous frames with the same value
#     #     def group_intervals(history):
#     #         intervals = []
#     #         if not history:
#     #             return intervals
#     #         prev_frame = history[0][0]
#     #         start, current_val = history[0]
#     #         for frame, val in history[1:]:
#     #             if val != current_val:
#     #                 intervals.append({'start': start, 'end': prev_frame, 'value': current_val})
#     #                 start, current_val = frame, val
#     #             prev_frame = frame
#     #         intervals.append({'start': start, 'end': prev_frame, 'value': current_val})
#     #         return intervals
#     #
#     #     # Turn each attribute’s per-frame list into contiguous intervals
#     #     for ent_id, data in entity_data.items():
#     #         for attr in attributes_to_use:
#     #             data[attr] = group_intervals(data['_history'][attr])
#     #         data.pop('_history', None)
#     #
#     #     return entity_data
#     def extract_entity_time_slots(self, attributes_to_use=None, entity_type="human"):
#         """
#         Group frame-by-frame values (including HHI/HOI interactions) into contiguous time slots per entity.
#         Supports node attributes (e.g., action) and edge-based interactions:
#           - "hhi": human-human interactions (physical edges), returns list of (other_id, relation)
#           - "hoi":  human-object interactions (physical edges), returns list of (object_id, relation)
#         """
#         if attributes_to_use is None:
#             attributes_to_use = ["action"]
#
#         # 1) sort frame IDs numerically
#         frame_ids = sorted(
#             self.frame_data.keys(),
#             key=lambda x: int(x.split('.')[0])
#         )
#
#         # 2) prepare storage
#         entity_data = {}
#
#         # 3) collect each attribute per frame
#         for frame_id in frame_ids:
#             frame_num = int(frame_id.split('.')[0])
#
#             for node_key, data in self.frame_data[frame_id].items():
#                 if data.get('type') != entity_type:
#                     continue
#
#                 # use whatever you stored as 'ID' (e.g. "pedestrian:6") as the entity key
#                 entity_id = str(data.get('ID', node_key))
#                 if entity_id not in entity_data:
#                     entity_data[entity_id] = {attr: [] for attr in attributes_to_use}
#                     entity_data[entity_id]['_history'] = {attr: [] for attr in attributes_to_use}
#
#                 # rebuild the exact graph node name used by add_human:
#                 #   node_key is "h_<human_id>"
#                 #   so human_id = node_key.split('_',1)[1]
#                 human_id = node_key.split('_', 1)[1]
#                 src = f"h_{frame_id}_{human_id}"
#
#                 for attr in attributes_to_use:
#                     if attr == "hhi":
#                         interactions = []
#                         # Debug:
#                         print("SRC:", src,
#                               "NODES:", [n for n in self.graph.nodes() if n.startswith(f"h_{frame_id}_")],
#                               "EDGES:", list(self.graph.out_edges(src, data=True, keys=True)))
#
#                         for _, tgt, _, e in self.graph.out_edges(src, data=True, keys=True):
#                             if e.get('edge_type') == 'physical' and self.graph.nodes[tgt]['node_type'] == 'human':
#                                 # Option A: store the full node name
#                                 interactions.append((tgt, e['relation_type']))
#
#                                 # Or Option B: extract just the integer
#                                 # other_int = tgt.split('_')[-1]
#                                 # interactions.append((other_int, e['relation_type']))
#
#                         val = tuple(sorted(interactions, key=lambda x: x[0])) if interactions else None
#
#
#                     elif attr == "hoi":
#                         interactions = []
#                         for _, tgt, _, edge_data in self.graph.out_edges(src, data=True, keys=True):
#                             if (edge_data.get('edge_type') == 'physical'
#                                     and self.graph.nodes[tgt].get('node_type') == 'object'):
#                                 obj_id = self.graph.nodes[tgt]['ID']
#                                 rel = edge_data.get('relation_type')
#                                 interactions.append((obj_id, rel))
#                         val = tuple(sorted(interactions, key=lambda x: x[0])) if interactions else None
#
#                     else:
#                         # any other node attribute (e.g. 'action', 'age', etc.)
#                         val = data.get(attr)
#
#                     entity_data[entity_id]['_history'][attr].append((frame_num, val))
#
#         # 4) helper to merge contiguous identical values into intervals
#         def group_intervals(history):
#             if not history:
#                 return []
#             intervals = []
#             prev_frame = history[0][0]
#             start, current_val = history[0]
#             for frame, val in history[1:]:
#                 if val != current_val:
#                     intervals.append({
#                         'start': start,
#                         'end': prev_frame,
#                         'value': current_val
#                     })
#                     start, current_val = frame, val
#                 prev_frame = frame
#             intervals.append({
#                 'start': start,
#                 'end': prev_frame,
#                 'value': current_val
#             })
#             return intervals
#
#         # 5) build final output: replace histories with intervals
#         for ent_id, data in entity_data.items():
#             for attr in attributes_to_use:
#                 data[attr] = group_intervals(data['_history'][attr])
#             data.pop('_history', None)
#
#         return entity_data
#
#     def get_frame_entities(self,
#                            frame_id: str,
#                            entity_type: str = None,
#                            attribute_filters: dict = None,
#                            return_fields: list = None
#                            ) -> dict:
#         """
#         For a given frame, return whatever node‐level fields you like,
#         optionally filtering by type and/or other attributes.
#
#         Args:
#             frame_id:          e.g. "0001"
#             entity_type:       if set, only nodes with data['type']==entity_type
#                                (e.g. 'human' or 'object') are returned
#             attribute_filters: dict of {attr_name: value_or_collection}
#                                to pre‐filter nodes (same logic as before)
#             return_fields:     list of the data keys you want back
#                                (e.g. ['bbox','age','action','gender']).
#                                If None, returns *all* stored fields.
#
#         Returns:
#             { node_key: { field: value, … }, … }
#         """
#         results = {}
#         nodes = self.frame_data.get(frame_id, {})
#
#         for node_key, data in nodes.items():
#             # 1) filter by type
#             if entity_type and data.get('type') != entity_type:
#                 continue
#
#             # 2) filter by arbitrary attribute values
#             if attribute_filters:
#                 ok = True
#                 for attr, required in attribute_filters.items():
#                     val = data.get(attr)
#                     if isinstance(required, (list, tuple, set)):
#                         if val not in required:
#                             ok = False;
#                             break
#                     else:
#                         if val != required:
#                             ok = False;
#                             break
#                 if not ok:
#                     continue
#
#             # 3) build the output record
#             if return_fields is None:
#                 # copy everything
#                 record = data.copy()
#             else:
#                 record = {f: data.get(f) for f in return_fields}
#
#             results[node_key] = record
#
#         return results
#
#         # --- New search helper method ---
#
#     def _search_graph(self, attr_dict, return_fields, time_window):
#         """
#         Search the scene graph for nodes matching all attributes in `attr_dict`
#         within the specified [start, end] time window.
#
#         Args:
#             attr_dict:      dict like {'age': 'young', 'gender': 'female', 'node_type': 'human'}
#             return_fields:  list of fields to return per result, e.g. ['bbox']
#             time_window:    (start_frame, end_frame)
#
#         Returns:
#             List of dicts with image_id, frame, node_name, and the requested fields.
#         """
#         start, end = time_window
#         results = []
#
#         # Handle alias: 'node_type' -> 'type'
#         attr_dict = {
#             ('type' if k == 'node_type' else k): v
#             for k, v in attr_dict.items()
#         }
#
#         for frame_id, nodes in self.frame_data.items():
#             try:
#                 frame_num = int(frame_id.split('.')[0])
#             except ValueError:
#                 continue
#             if not (start <= frame_num <= end):
#                 continue
#
#             for node_key, data in nodes.items():
#                 # Check all attribute conditions
#                 if all(data.get(k) == v for k, v in attr_dict.items()):
#                     # Build the result entry, including node_name
#                     entry = {
#                         'image_id': frame_id,
#                         'frame': frame_num,
#                         'node_name': node_key
#                     }
#                     # Add any requested fields
#                     for field in return_fields:
#                         entry[field] = data.get(field)
#
#                     results.append(entry)
#
#         # Sort results by frame number
#         results.sort(key=lambda x: x['frame'])
#         return results
#
#
import matplotlib.pyplot as plt    
import networkx as nx

class SpatialTemporalSceneGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()  # allows multiple edges between nodes
        self.frame_data = {}            # Stores data per frame

    def add_human(self, frame_id, age, race, gender, action,
                  social_group_id, group_size, bbox, bbox_3d,
                  human_id_in_frame, occlusion,
                  SR_Robot_Ref=None, distance=None,face_visibility=None):
        """
        Add a human node with optional spatial-reference to robot and distance.
        """
        if frame_id not in self.frame_data:
            self.frame_data[frame_id] = {}
        key = f"h_{human_id_in_frame}"
        self.frame_data[frame_id][key] = {
            'type': 'human', 'age': age, 'race': race,
            'gender': gender, 'action': action,
            'social_group_id': social_group_id,
            'group_size': group_size, 'bbox': bbox,
            'bbox_3d': bbox_3d,  # added here
            'occlusion': occlusion,
            'SR_Robot_Ref': SR_Robot_Ref,
            'distance': distance,
            'face_visibility': face_visibility

        }
        node_name = key
        self.graph.add_node(
            node_name,
            node_type='human', age=age, race=race,
            gender=gender, action=action,
            social_group_id=social_group_id,
            group_size=group_size, bbox=bbox,
            bbox_3d=bbox_3d,  # added here
            occlusion=occlusion,
            SR_Robot_Ref=SR_Robot_Ref,
            distance=distance,
            face_visibility=face_visibility,
            ID=human_id_in_frame
        )

    def add_object(self, frame_id, obj_name, tracking_id, category_id, bbox=None):
        if frame_id not in self.frame_data:
            self.frame_data[frame_id] = {}
        key = f"o_{tracking_id}_{category_id}"
        self.frame_data[frame_id][key] = {
            'type': 'object', 'obj_name': obj_name,
            'tracking_id': tracking_id, 'category_id': category_id,
            'bbox': bbox
        }
        node_name = key
        self.graph.add_node(node_name,
                            node_type='object', obj_name=obj_name,
                            tracking_id=tracking_id, category_id=category_id,
                            bbox=bbox, ID=f"{tracking_id}_{category_id}")

    def add_physical_relationship(self, frame_id, node_key_1, node_key_2, relation_type):
        self.graph.add_edge(
            node_key_1,
            node_key_2,
            relation_type=relation_type,
            edge_type='physical'
        )

    def add_geometric_relationship(self, frame_id, node_key_1, node_key_2, relation_type):
        self.graph.add_edge(
            node_key_1,
            node_key_2,
            relation_type=relation_type,
            edge_type='geometric'
        )

    # def extract_entity_time_slots(self, attributes_to_use=None, entity_type="human"):
    #     """
    #     Group frame-by-frame values (including HHI/HOI) into contiguous time slots per entity.
    #     """
    #     if attributes_to_use is None:
    #         attributes_to_use = ["action"]
    #
    #     # sort frames numerically if possible
    #     try:
    #         frame_ids = sorted(
    #             self.frame_data.keys(),
    #             key=lambda x: int(x.split('.')[0])
    #         )
    #     except Exception:
    #         frame_ids = list(self.frame_data.keys())
    #
    #     entity_data = {}
    #
    #     for frame_id in frame_ids:
    #         for node_key, data in self.frame_data[frame_id].items():
    #             if data.get('type') != entity_type:
    #                 continue
    #
    #             entity_id = str(data.get('ID', node_key))
    #             if entity_id not in entity_data:
    #                 entity_data[entity_id] = {attr: [] for attr in attributes_to_use}
    #                 entity_data[entity_id]['_history'] = {attr: [] for attr in attributes_to_use}
    #
    #             src = node_key
    #             for attr in attributes_to_use:
    #                 if attr == "hhi":
    #                     interactions = []
    #                     for _, tgt, _, e in self.graph.out_edges(src, data=True, keys=True):
    #                         if e.get('edge_type') == 'physical' and \
    #                                 self.graph.nodes[tgt].get('node_type') == 'human':
    #
    #                             relation = e.get('relation_type')
    #                             tgt_data = self.frame_data[frame_id].get(tgt, {})
    #                             bbox = tgt_data.get('bbox')
    #                             gender = tgt_data.get('gender')
    #                             age = tgt_data.get('age')
    #                             race = tgt_data.get('race')
    #
    #                             if bbox and relation:
    #                                 interactions.append({
    #                                     'interactions': [relation],
    #                                     'box_pair': bbox,
    #                                     'gender_pair': gender,
    #                                     'age_pair': age,
    #                                     'race_pair': race,
    #                                     'id_pair': tgt,
    #                                     'type': 'specific'
    #                                 })
    #                     val = interactions if interactions else None
    #
    #                 elif attr == "hoi":
    #                     interactions = set()
    #                     for _, tgt, _, e in self.graph.out_edges(src, data=True, keys=True):
    #                         if e.get('edge_type') == 'physical' and \
    #                                 self.graph.nodes[tgt].get('node_type') == 'object':
    #                             interactions.add((tgt, e.get('relation_type')))
    #                     val = tuple(sorted(interactions, key=lambda x: x[0])) if interactions else None
    #
    #                 elif attr == "hhg":
    #                     # New logic for "hhg" - geometric edge relation, formatted like hhi
    #                     interactions = []
    #                     for _, tgt, _, e in self.graph.out_edges(src, data=True, keys=True):
    #                         if e.get('edge_type') == 'geometric':  # Look for geometric edges
    #                             relation = e.get('relation_type')
    #                             # geometric_relation = e.get('geometry', {}).get('relation')  # Geometric relation
    #                             if relation:
    #                                 tgt_data = self.frame_data[frame_id].get(tgt, {})
    #                                 bbox = tgt_data.get('bbox')
    #                                 gender = tgt_data.get('gender')
    #                                 age = tgt_data.get('age')
    #                                 race = tgt_data.get('race')
    #
    #                                 if bbox:
    #                                     interactions.append({
    #                                         'interactions': [relation],
    #                                         'box_pair': bbox,
    #                                         'gender_pair': gender,
    #                                         'age_pair': age,
    #                                         'race_pair': race,
    #                                         'id_pair': tgt,
    #                                         'type': 'specific'
    #                                     })
    #                     val = interactions if interactions else None
    #
    #                 else:
    #                     val = data.get(attr)
    #
    #                 try:
    #                     frame_num = int(frame_id.split('.')[0])
    #                 except Exception:
    #                     frame_num = frame_id
    #                 entity_data[entity_id]['_history'][attr].append((frame_num, val))
    #
    #     def group_intervals(history):
    #         if not history:
    #             return []
    #         intervals = []
    #         prev_frame = history[0][0]
    #         start, current_val = history[0]
    #         for frame, val in history[1:]:
    #             if val != current_val:
    #                 intervals.append({'start': start, 'end': prev_frame, 'value': current_val})
    #                 start, current_val = frame, val
    #             prev_frame = frame
    #         intervals.append({'start': start, 'end': prev_frame, 'value': current_val})
    #         return intervals
    #
    #     for ent_id, data in entity_data.items():
    #         for attr in attributes_to_use:
    #             intervals = group_intervals(data['_history'][attr])
    #             # merge multiple HHI entries into single dict with aggregated interactions
    #             if attr == 'hhi':
    #                 merged = []
    #                 for interval in intervals:
    #                     vals = interval['value'] or []
    #                     if not vals:
    #                         merged.append(interval)
    #                         continue
    #                     # merge dicts
    #                     base = vals[0].copy()
    #                     all_rels = []
    #                     for d in vals:
    #                         all_rels.extend(d.get('interactions', []))
    #                     # dedupe in order
    #                     seen = set()
    #                     uniq = []
    #                     for rel in all_rels:
    #                         if rel not in seen:
    #                             seen.add(rel)
    #                             uniq.append(rel)
    #                     base['interactions'] = uniq
    #                     interval['value'] = base
    #                     merged.append(interval)
    #                 data[attr] = merged
    #             elif attr == 'hhg':
    #                 merged = []
    #                 for interval in intervals:
    #                     vals = interval['value'] or []
    #                     if not vals:
    #                         merged.append(interval)
    #                         continue
    #                     # merge dicts
    #                     base = vals[0].copy()
    #                     all_rels = []
    #                     for d in vals:
    #                         all_rels.extend(d.get('interactions', []))
    #                     # dedupe in order
    #                     seen = set()
    #                     uniq = []
    #                     for rel in all_rels:
    #                         if rel not in seen:
    #                             seen.add(rel)
    #                             uniq.append(rel)
    #                     base['interactions'] = uniq
    #                     interval['value'] = base
    #                     merged.append(interval)
    #                 data[attr] = merged
    #             else:
    #                 data[attr] = intervals
    #         data.pop('_history', None)
    #
    #     return entity_data
    # def extract_entity_time_slots(self, attributes_to_use=None, entity_type="human"):
    #     """
    #     Group frame-by-frame values (including HHI/HHG/HOI) into contiguous time slots per entity.
    #     """
    #     if attributes_to_use is None:
    #         attributes_to_use = ["action"]
    #
    #     # Sort frame IDs numerically if possible
    #     try:
    #         frame_ids = sorted(
    #             self.frame_data.keys(),
    #             key=lambda x: int(x.split('.')[0])
    #         )
    #     except Exception:
    #         frame_ids = list(self.frame_data.keys())
    #
    #     entity_data = {}
    #
    #     for frame_id in frame_ids:
    #         for node_key, data in self.frame_data[frame_id].items():
    #             if data.get('type') != entity_type:
    #                 continue
    #
    #             entity_id = str(data.get('ID', node_key))
    #             if entity_id not in entity_data:
    #                 entity_data[entity_id] = {attr: [] for attr in attributes_to_use}
    #                 entity_data[entity_id]['_history'] = {attr: [] for attr in attributes_to_use}
    #
    #             src = node_key
    #             for attr in attributes_to_use:
    #                 if attr in ("hhi", "hhg"):
    #                     interactions = []
    #                     edge_type = 'physical' if attr == "hhi" else 'geometric'
    #
    #                     for _, tgt, _, e in self.graph.out_edges(src, data=True, keys=True):
    #                         if e.get('edge_type') == edge_type and \
    #                                 self.graph.nodes[tgt].get('node_type') == 'human':
    #
    #                             relation = e.get('relation_type')
    #                             tgt_data = self.frame_data[frame_id].get(tgt, {})
    #                             bbox = tgt_data.get('bbox')
    #                             gender = tgt_data.get('gender')
    #                             age = tgt_data.get('age')
    #                             race = tgt_data.get('race')
    #
    #                             if bbox and relation:
    #                                 # Add interaction similar to hhi, distinguishing with edge_type
    #                                 interaction = {
    #                                     'interactions': [relation],  # List of interactions
    #                                     # 'box_pair': bbox,
    #                                     'gender_pair': gender,
    #                                     'age_pair': age,
    #                                     'race_pair': race,
    #                                     'id_pair': tgt,
    #                                     'type': 'geometric' if attr == 'hhg' else 'physical'
    #                                 }
    #                                 if interaction not in interactions:
    #                                    interactions.append(interaction)
    #
    #                     # If there are interactions, we take the first one; otherwise, None
    #                     val = interactions if interactions else None
    #
    #                 elif attr == "hoi":
    #                     interactions = set()
    #                     for _, tgt, _, e in self.graph.out_edges(src, data=True, keys=True):
    #                         if e.get('edge_type') == 'physical' and \
    #                                 self.graph.nodes[tgt].get('node_type') == 'object':
    #                             interactions.add((tgt, e.get('relation_type')))
    #                     val = tuple(sorted(interactions, key=lambda x: x[0])) if interactions else None
    #                 else:
    #                     val = data.get(attr)
    #
    #                 try:
    #                     frame_num = int(frame_id.split('.')[0])
    #                 except Exception:
    #                     frame_num = frame_id
    #                 entity_data[entity_id]['_history'][attr].append((frame_num, val))
    #
    #     # def group_intervals(history):
    #     #     if not history:
    #     #         return []
    #     #     intervals = []
    #     #     prev_frame = history[0][0]
    #     #     start, current_val = history[0]
    #     #     for frame, val in history[1:]:
    #     #         if val != current_val:
    #     #             intervals.append({'start': start, 'end': prev_frame, 'value': current_val})
    #     #             start, current_val = frame, val
    #     #         prev_frame = frame
    #     #     intervals.append({'start': start, 'end': prev_frame, 'value': current_val})
    #     #     return intervals
    #     def group_intervals(history):
    #         from collections import defaultdict
    #
    #         # Dictionary to track current intervals by a unique key
    #         active_intervals = {}
    #         results = []
    #
    #         for frame, events in history:
    #             current_keys = set()
    #
    #             if not events:
    #                 return []  # or handle appropriately
    #
    #             for event in events:
    #                 print(events)
    #                 print(event)
    #                 key = (
    #                     tuple(event['interactions']),
    #                     event['gender_pair'],
    #                     event['age_pair'],
    #                     event['race_pair'],
    #                     event['id_pair'],
    #                     event['type']
    #                 )
    #                 current_keys.add(key)
    #
    #                 if key in active_intervals:
    #                     active_intervals[key]['end'] = frame
    #                 else:
    #                     # Start a new interval
    #                     active_intervals[key] = {
    #                         'start': frame,
    #                         'end': frame,
    #                         'value': {
    #                             'interactions': event['interactions'],
    #                             'gender_pair': event['gender_pair'],
    #                             'age_pair': event['age_pair'],
    #                             'race_pair': event['race_pair'],
    #                             'id_pair': event['id_pair'],
    #                             'type': event['type']
    #                         }
    #                     }
    #
    #             # End intervals that are no longer present
    #             to_delete = []
    #             for key in active_intervals:
    #                 if key not in current_keys:
    #                     results.append(active_intervals[key])
    #                     to_delete.append(key)
    #             for key in to_delete:
    #                 del active_intervals[key]
    #
    #         # Add remaining active intervals
    #         results.extend(active_intervals.values())
    #         return results
    #
    #     for ent_id, data in entity_data.items():
    #         for attr in attributes_to_use:
    #             intervals = group_intervals(data['_history'][attr])
    #
    #             # if attr in ('hhi', 'hhg'):
    #             #     merged = []
    #             #     for interval in intervals:
    #             #         vals = interval['value'] or []
    #             #         if not vals:  # If the list is empty, just add the interval as it is
    #             #             merged.append(interval)
    #             #             continue
    #             #
    #             #         # For both hhi and hhg, handle them similarly
    #             #         if len(vals) > 0:
    #             #             base = vals[0]  # We take the first valid interaction (either hhi or hhg)
    #             #             interval['value'] = base
    #             #         merged.append(interval)
    #             #
    #             #     data[attr] = merged
    #             # else:
    #             #     data[attr] = intervals
    #             if attr in ('hhi', 'hhg'):
    #                 merged = []
    #                 for interval in intervals:
    #                     val = interval['value']
    #                     if not val:  # If empty or None
    #                         merged.append(interval)
    #                         continue
    #
    #                     # No need to extract base — it's already the dict
    #                     interval['value'] = val
    #                     merged.append(interval)
    #
    #                 data[attr] = merged
    #             else:
    #                 data[attr] = intervals
    #
    #         data.pop('_history', None)
    #
    #     return entity_data
    # def extract_entity_time_slots(self, attributes_to_use=None, entity_type="human"):
    #     """
    #     Group frame-by-frame values (scalar and interactions) into contiguous time slots per entity.
    #
    #     :param attributes_to_use: list of attributes (e.g., ['age', 'race', 'gender', 'action', 'hhi', 'hhg', 'hoi'])
    #     :param entity_type: filter nodes by this type (default 'human')
    #     :return: dict mapping entity_id to dict of attribute intervals
    #     """
    #     # Default to action if no attributes specified
    #     if attributes_to_use is None:
    #         attributes_to_use = ['action']
    #
    #     # Sort frame IDs numerically if possible
    #     try:
    #         frame_ids = sorted(
    #             self.frame_data.keys(),
    #             key=lambda x: int(x.split('.')[0])
    #         )
    #     except Exception:
    #         frame_ids = list(self.frame_data.keys())
    #
    #     # Initialize container for per-entity histories
    #     entity_data = {}
    #     for frame_id in frame_ids:
    #         try:
    #             frame_num = int(frame_id.split('.')[0])
    #         except ValueError:
    #             frame_num = frame_id
    #
    #         for node_key, data in self.frame_data[frame_id].items():
    #             if data.get('type') != entity_type:
    #                 continue
    #
    #             entity_id = str(data.get('ID', node_key))
    #             if entity_id not in entity_data:
    #                 # prepare history lists for each attribute
    #                 entity_data[entity_id] = {
    #                     attr: [] for attr in attributes_to_use
    #                 }
    #                 entity_data[entity_id]['_history'] = {
    #                     attr: [] for attr in attributes_to_use
    #                 }
    #
    #             # Gather frame-by-frame values or event lists
    #             for attr in attributes_to_use:
    #                 if attr in ('hhi', 'hhg'):
    #                     # collect human interaction events
    #                     events = []
    #                     edge_type = 'physical' if attr == 'hhi' else 'geometric'
    #                     for _, tgt, _, e in self.graph.out_edges(node_key, data=True, keys=True):
    #                         if (e.get('edge_type') == edge_type and
    #                                 self.graph.nodes[tgt].get('node_type') == 'human'):
    #                             relation = e.get('relation_type')
    #                             tgt_data = self.frame_data[frame_id].get(tgt, {})
    #                             bbox = tgt_data.get('bbox')
    #                             if bbox and relation:
    #                                 events.append({
    #                                     'interactions': [relation],
    #                                     'gender_pair': tgt_data.get('gender'),
    #                                     'age_pair': tgt_data.get('age'),
    #                                     'race_pair': tgt_data.get('race'),
    #                                     'id_pair': tgt,
    #                                     'type': attr
    #                                 })
    #                     val = events if events else None
    #
    #                 elif attr == 'hoi':  # human-object interactions
    #                     interactions = set()
    #                     for _, tgt, _, e in self.graph.out_edges(node_key, data=True, keys=True):
    #                         if (e.get('edge_type') == 'physical' and
    #                                 self.graph.nodes[tgt].get('node_type') == 'object'):
    #                             interactions.add((tgt, e.get('relation_type')))
    #                     val = tuple(sorted(interactions, key=lambda x: x[0])) if interactions else None
    #
    #                 else:
    #                     # scalar attribute: age, race, gender, action, etc.
    #                     val = data.get(attr)
    #
    #                 # Append to history
    #                 entity_data[entity_id]['_history'][attr].append((frame_num, val))
    #
    #     # Helper: group contiguous scalar values
    #     def group_scalar_intervals(history):
    #         if not history:
    #             return []
    #         intervals = []
    #         start, prev_val = history[0]
    #         prev_frame = start
    #         for frame, val in history[1:]:
    #             if val != prev_val:
    #                 intervals.append({'start': start, 'end': prev_frame, 'value': prev_val})
    #                 start, prev_val = frame, val
    #             prev_frame = frame
    #         intervals.append({'start': start, 'end': prev_frame, 'value': prev_val})
    #         return intervals
    #
    #     # Helper: group event-list intervals
    #     def group_event_intervals(history):
    #         active, results = {}, []
    #         for frame, events in history:
    #             # if no events, close all open intervals
    #             if not events:
    #                 results.extend(active.values())
    #                 active.clear()
    #                 continue
    #
    #             current_keys = set()
    #             for ev in events:
    #                 key = (
    #                     tuple(ev['interactions']),
    #                     ev['gender_pair'], ev['age_pair'], ev['race_pair'],
    #                     ev['id_pair'], ev['type']
    #                 )
    #                 current_keys.add(key)
    #                 if key in active:
    #                     active[key]['end'] = frame
    #                 else:
    #                     active[key] = {'start': frame, 'end': frame, 'value': ev}
    #
    #             # close intervals for missing keys
    #             for key in set(active) - current_keys:
    #                 results.append(active.pop(key))
    #
    #         # flush remainders
    #         results.extend(active.values())
    #         return results
    #
    #     # Convert histories to intervals per attribute
    #     for ent_id, data in entity_data.items():
    #         for attr in attributes_to_use:
    #             hist = data['_history'][attr]
    #             if attr in ('hhi', 'hhg', 'hoi'):
    #                 data[attr] = group_event_intervals(hist)
    #             else:
    #                 data[attr] = group_scalar_intervals(hist)
    #         # remove raw history
    #         data.pop('_history', None)
    #
    #     return entity_data
    # def extract_entity_time_slots(self, attributes_to_use=None, entity_type="human"):
    #     """
    #     Group frame-by-frame values (scalar and interactions) into contiguous time slots per entity.
    #
    #     :param attributes_to_use: list of attributes (e.g., ['age', 'race', 'gender', 'action', 'hhi', 'hhg', 'hoi'])
    #     :param entity_type: filter nodes by this type (default 'human')
    #     :return: dict mapping entity_id to dict of attribute intervals
    #     """
    #     if attributes_to_use is None:
    #         attributes_to_use = ['action']
    #
    #     try:
    #         frame_ids = sorted(self.frame_data.keys(), key=lambda x: int(x.split('.')[0]))
    #     except Exception:
    #         frame_ids = list(self.frame_data.keys())
    #
    #     entity_data = {}
    #     for frame_id in frame_ids:
    #         try:
    #             frame_num = int(frame_id.split('.')[0])
    #         except ValueError:
    #             frame_num = frame_id
    #
    #         for node_key, data in self.frame_data[frame_id].items():
    #             if data.get('type') != entity_type:
    #                 continue
    #
    #             entity_id = str(data.get('ID', node_key))
    #             if entity_id not in entity_data:
    #                 entity_data[entity_id] = {
    #                     attr: [] for attr in attributes_to_use
    #                 }
    #                 entity_data[entity_id]['_history'] = {
    #                     attr: [] for attr in attributes_to_use
    #                 }
    #
    #             for attr in attributes_to_use:
    #                 val = None
    #
    #                 if attr in ('hhi', 'hhg', 'hoi'):
    #                     events = []
    #                     is_human_interaction = attr in ('hhi', 'hhg')
    #                     edge_type = 'geometric' if attr == 'hhg' else 'physical'
    #
    #                     for _, tgt, _, e in self.graph.out_edges(node_key, data=True, keys=True):
    #                         tgt_type = self.graph.nodes[tgt].get('node_type')
    #                         if e.get('edge_type') != edge_type:
    #                             continue
    #
    #                         if is_human_interaction and tgt_type == 'human':
    #                             relation = e.get('relation_type')
    #                             tgt_data = self.frame_data[frame_id].get(tgt, {})
    #                             if relation:
    #                                 events.append({
    #                                     'interactions': [relation],
    #                                     'gender_pair': tgt_data.get('gender'),
    #                                     'age_pair': tgt_data.get('age'),
    #                                     'race_pair': tgt_data.get('race'),
    #                                     'id_pair': tgt,
    #                                     'type': attr
    #                                 })
    #                         elif attr == 'hoi' and tgt_type == 'object':
    #                             relation = e.get('relation_type')
    #                             tgt_data = self.frame_data[frame_id].get(tgt, {})
    #                             obj_name = tgt_data.get('obj_name')
    #
    #                             try:
    #                                 _, track_id, cat_id = tgt.split('_', 2)
    #                             except ValueError:
    #                                 track_id, cat_id = None, None
    #
    #                             if relation and obj_name:
    #                                 events.append({
    #                                     'interactions': [relation],
    #                                     'obj_name': obj_name,
    #                                     'id': tgt,
    #                                     'tracking_id': track_id,
    #                                     'category_id': cat_id,
    #                                     'type': 'hoi'
    #                                 })
    #                     val = events if events else None
    #                 else:
    #                     val = data.get(attr)
    #
    #                 entity_data[entity_id]['_history'][attr].append((frame_num, val))
    #
    #     def group_scalar_intervals(history):
    #         if not history:
    #             return []
    #         intervals = []
    #         start, prev_val = history[0]
    #         prev_frame = start
    #         for frame, val in history[1:]:
    #             if val != prev_val:
    #                 intervals.append({'start': start, 'end': prev_frame, 'value': prev_val})
    #                 start, prev_val = frame, val
    #             prev_frame = frame
    #         intervals.append({'start': start, 'end': prev_frame, 'value': prev_val})
    #         return intervals
    #
    #     def group_event_intervals(history):
    #         active, results = {}, []
    #         for frame, events in history:
    #             if not events:
    #                 results.extend(active.values())
    #                 active.clear()
    #                 continue
    #
    #             current_keys = set()
    #             for ev in events:
    #                 key = (
    #                     tuple(ev.get('interactions', [])),
    #                     ev.get('gender_pair'), ev.get('age_pair'), ev.get('race_pair'),
    #                     ev.get('id_pair'), ev.get('type'),
    #                     ev.get('obj_name'), ev.get('tracking_id'), ev.get('category_id')
    #                 )
    #                 current_keys.add(key)
    #                 if key in active:
    #                     active[key]['end'] = frame
    #                 else:
    #                     active[key] = {'start': frame, 'end': frame, 'value': ev}
    #
    #             closed_keys = set(active.keys()) - current_keys
    #             for key in closed_keys:
    #                 results.append(active.pop(key))
    #
    #         results.extend(active.values())
    #         return results
    #
    #     for ent_id, data in entity_data.items():
    #         for attr in attributes_to_use:
    #             hist = data['_history'][attr]
    #             if attr in ('hhi', 'hhg', 'hoi'):
    #                 data[attr] = group_event_intervals(hist)
    #             else:
    #                 data[attr] = group_scalar_intervals(hist)
    #         del data['_history']
    #
    #     return entity_data
    def extract_entity_time_slots(self, attributes_to_use=None, entity_type="human"):
        """
        Group frame-by-frame values (scalar and interactions) into contiguous time slots per entity.

        :param attributes_to_use: list of attributes (e.g., ['age', 'race', 'gender', 'action', 'hhi', 'hhg', 'hoi'])
        :param entity_type: filter nodes by this type (default 'human')
        :return: dict mapping entity_id to dict of attribute intervals
        """
        if attributes_to_use is None:
            attributes_to_use = ['action']

        try:
            frame_ids = sorted(self.frame_data.keys(), key=lambda x: int(x.split('.')[0]))
        except Exception:
            frame_ids = list(self.frame_data.keys())

        entity_data = {}
        for frame_id in frame_ids:
            try:
                frame_num = int(frame_id.split('.')[0])
            except ValueError:
                frame_num = frame_id

            for node_key, data in self.frame_data[frame_id].items():
                # if data.get('type') != entity_type:
                #     continue
                node_type = data.get('type')
                if entity_type == 'object' and node_type != 'object':
                    continue
                elif entity_type in ('human', 'human&object') and node_type != 'human':
                    continue

                entity_id = str(data.get('ID', node_key))
                if entity_id not in entity_data:
                    entity_data[entity_id] = {attr: [] for attr in attributes_to_use}
                    entity_data[entity_id]['_history'] = {attr: [] for attr in attributes_to_use}

                for attr in attributes_to_use:
                    val = None

                    if attr in ('hhi', 'hhg', 'hoi'):
                        events = []
                        is_human_interaction = attr in ('hhi', 'hhg')
                        edge_type = 'geometric' if attr == 'hhg' else 'physical'

                        for _, tgt, _, e in self.graph.out_edges(node_key, data=True, keys=True):
                            tgt_type = self.graph.nodes[tgt].get('node_type')
                            if e.get('edge_type') != edge_type:
                                continue

                            if is_human_interaction and tgt_type == 'human':
                                relation = e.get('relation_type')
                                tgt_data = self.frame_data[frame_id].get(tgt, {})
                                if relation:
                                    events.append({
                                        'interactions': [relation],
                                        'gender_pair': tgt_data.get('gender'),
                                        'age_pair': tgt_data.get('age'),
                                        'race_pair': tgt_data.get('race'),
                                        'id_pair': tgt,
                                        'type': attr
                                    })
                            elif attr == 'hoi' and tgt_type == 'object':
                                relation = e.get('relation_type')
                                tgt_data = self.frame_data[frame_id].get(tgt, {})
                                obj_name = tgt_data.get('obj_name')
                                track_id = tgt_data.get('tracking_id')
                                cat_id = tgt_data.get('category_id')
                                if relation:
                                    events.append({
                                        'interactions': [relation],
                                        'obj_name': obj_name,
                                        'id': tgt,
                                        'tracking_id': track_id,
                                        'category_id': cat_id,
                                        'type': 'hoi'
                                    })

                        val = events
                    else:
                        # scalar attribute (e.g., action, age, race, gender)
                        val = data.get(attr)

                    # record history for interval grouping
                    entity_data[entity_id]['_history'][attr].append((frame_num, val))

        def group_scalar_intervals(history):
            if not history:
                return []
            intervals = []
            start, prev_val = history[0]
            prev_frame = start
            for frame, val in history[1:]:
                if val != prev_val:
                    intervals.append({'start': start, 'end': prev_frame, 'value': prev_val})
                    start, prev_val = frame, val
                prev_frame = frame
            intervals.append({'start': start, 'end': prev_frame, 'value': prev_val})
            return intervals

        def group_event_intervals(history):
            active, results = {}, []
            for frame, events in history:
                if not events:
                    results.extend(active.values())
                    active.clear()
                    continue

                current_keys = set()
                for ev in events:
                    key = (
                        tuple(ev.get('interactions', [])),
                        ev.get('gender_pair'), ev.get('age_pair'), ev.get('race_pair'),
                        ev.get('id_pair'), ev.get('type'),
                        ev.get('obj_name'), ev.get('tracking_id'), ev.get('category_id')
                    )
                    current_keys.add(key)
                    if key in active:
                        active[key]['end'] = frame
                    else:
                        active[key] = {'start': frame, 'end': frame, 'value': ev}

                closed_keys = set(active.keys()) - current_keys
                for key in closed_keys:
                    results.append(active.pop(key))

            results.extend(active.values())
            return results

        for ent_id, data in entity_data.items():
            for attr in attributes_to_use:
                hist = data['_history'][attr]
                if attr in ('hhi', 'hhg', 'hoi'):
                    data[attr] = group_event_intervals(hist)
                else:
                    data[attr] = group_scalar_intervals(hist)
            del data['_history']

        return entity_data

    def get_frame_entities(self, frame_id, entity_type=None,
                           attribute_filters=None, return_fields=None):
        """ Return nodes in a frame with optional filtering. """
        results = {}
        nodes = self.frame_data.get(frame_id, {})
        for node_key, data in nodes.items():
            if entity_type and data.get('type') != entity_type:
                continue
            if attribute_filters:
                if not all(
                    (data.get(k) in v) if isinstance(v, (list, tuple, set)) else data.get(k) == v
                    for k, v in attribute_filters.items()
                ):
                    continue
            record = data.copy() if return_fields is None else {f: data.get(f) for f in return_fields}
            results[node_key] = record
        return results

    # def _search_graph(self, attr_dict, return_fields, time_window):
    #     """Search nodes matching attributes in a frame window."""
    #     start, end = time_window
    #     results = []
    #     attr_dict = {('type' if k=='node_type' else k): v for k, v in attr_dict.items()}
    #     for frame_id, nodes in self.frame_data.items():
    #         try:
    #             frame_num = int(frame_id.split('.')[0])
    #         except:
    #             continue
    #         if not (start <= frame_num <= end):
    #             continue
    #         for node_key, data in nodes.items():
    #             if all(data.get(k) == v for k, v in attr_dict.items()):
    #                 entry = {'image_id': frame_id, 'frame': frame_num, 'node_name': node_key}
    #                 for f in return_fields:
    #                     entry[f] = data.get(f)
    #                 results.append(entry)
    #     results.sort(key=lambda x: x['frame'])
    #     return results

    # def _search_graph(self, attr_dict, return_fields, time_window):
    #     """
    #     Search scalar or interaction events in a frame window.
    #
    #     :param attr_dict: filters. For interactions include 'type':'hhi'|'hhg'|'hoi'
    #                       plus any of 'interactions','gender_pair','age_pair','race_pair'
    #                       (for hhi/hhg) or 'obj_name','tracking_id','category_id' (for hoi).
    #     :param return_fields: list of keys to return, e.g. ['frame','id','interactions',...].
    #     :param time_window: (start_frame, end_frame) inclusive.
    #     """
    #     # validate inputs
    #     if not isinstance(attr_dict, dict) \
    #             or not isinstance(return_fields, list) \
    #             or not isinstance(time_window, tuple):
    #         raise ValueError("Invalid attr_dict, return_fields, or time_window")
    #
    #     start_frame, end_frame = time_window
    #     search_type = attr_dict.get('type')
    #
    #     # identify remaining filters (beyond 'type')
    #     extra_filters = {k: v for k, v in attr_dict.items() if k != 'type'}
    #
    #     results = []
    #     for frame_id, nodes in self.frame_data.items():
    #         try:
    #             frame_num = int(frame_id.split('.')[0])
    #         except (ValueError, IndexError):
    #             continue
    #         if not (start_frame <= frame_num <= end_frame):
    #             continue
    #
    #         for node_key, data in nodes.items():
    #             # ------ SCALAR mode -------
    #             if search_type not in ('hhi', 'hhg', 'hoi'):
    #                 # handle shorthand {'type':'action','action':'sitting'}
    #                 filters = dict(attr_dict)
    #                 t = filters.pop('type', None)
    #                 if t and t not in ('human', 'object'):
    #                     # means t names the field to filter on
    #                     filters = {t: filters.get(t)}
    #                 elif t:
    #                     filters['type'] = t
    #                 # now drop 'type'
    #                 filters.pop('type', None)
    #
    #                 # node‐type filter
    #                 if 'type' in filters and data.get('type') != filters['type']:
    #                     continue
    #                 # other scalar filters
    #                 if any(data.get(k) != v for k, v in filters.items()):
    #                     continue
    #
    #                 # emit one record
    #                 rec = {'frame': frame_num}
    #                 for f in return_fields:
    #                     rec[f] = node_key if f == 'id' else data.get(f)
    #                 results.append(rec)
    #
    #             # ------ INTERACTION mode -------
    #             else:
    #                 # only humans can have hhi/hhg/hoi edges
    #                 if data.get('type') != 'human':
    #                     continue
    #
    #                 is_human_inter = search_type in ('hhi', 'hhg')
    #                 desired_edge = 'geometric' if search_type == 'hhg' else 'physical'
    #
    #                 # scan exactly the same edges as extract_entity_time_slots
    #                 for _, tgt, _, e in self.graph.out_edges(node_key, data=True, keys=True):
    #                     if e.get('edge_type') != desired_edge:
    #                         continue
    #
    #                     tgt_data = self.frame_data[frame_id].get(tgt, {})
    #                     tgt_type = tgt_data.get('type')
    #
    #                     # filter target type
    #                     if is_human_inter and tgt_type != 'human':
    #                         continue
    #                     if not is_human_inter and search_type == 'hoi' and tgt_type != 'object':
    #                         continue
    #
    #                     # build the event dict
    #                     ev = {
    #                         'frame': frame_num,
    #                         'id': node_key,
    #                         'interactions': e.get('relation_type'),
    #                         'id_pair': tgt,
    #                         'type': search_type
    #                     }
    #                     if is_human_inter:
    #                         ev.update({
    #                             'gender_pair': tgt_data.get('gender'),
    #                             'age_pair': tgt_data.get('age'),
    #                             'race_pair': tgt_data.get('race'),
    #                         })
    #                     else:  # hoi
    #                         ev.update({
    #                             'obj_name': tgt_data.get('obj_name'),
    #                             'tracking_id': tgt_data.get('tracking_id'),
    #                             'category_id': tgt_data.get('category_id'),
    #                         })
    #
    #                     # apply any extra_filters (e.g. interactions, gender_pair, etc.)
    #                     if any(ev.get(k) != v for k, v in extra_filters.items()):
    #                         continue
    #
    #                     # output only requested fields
    #                     rec = {}
    #                     for f in return_fields:
    #                         rec[f] = ev.get(f)
    #                     results.append(rec)
    #
    #     return results
    # def _search_graph(self, attr_dict, return_fields, time_window):
    #     """
    #     Search scalar or interaction events in a frame window.
    #
    #     :param attr_dict: filters. For interactions include 'type':'hhi'|'hhg'|'hoi'
    #                       plus any of 'interactions','gender_pair','age_pair','race_pair'
    #                       (for hhi/hhg) or 'obj_name','tracking_id','category_id' (for hoi).
    #     :param return_fields: list of keys to return, e.g. ['frame','id','interactions',...].
    #     :param time_window: (start_frame, end_frame) inclusive.
    #     """
    #     # validate inputs
    #     if not isinstance(attr_dict, dict) \
    #             or not isinstance(return_fields, list) \
    #             or not isinstance(time_window, tuple):
    #         raise ValueError("Invalid attr_dict, return_fields, or time_window")
    #
    #     start_frame, end_frame = time_window
    #     search_type = attr_dict.get('type')
    #
    #     # identify remaining filters (beyond 'type')
    #     extra_filters = {k: v for k, v in attr_dict.items() if k != 'type'}
    #
    #     results = []
    #     for frame_id, nodes in self.frame_data.items():
    #         try:
    #             frame_num = int(frame_id.split('.')[0])
    #         except (ValueError, IndexError):
    #             continue
    #         if not (start_frame <= frame_num <= end_frame):
    #             continue
    #
    #         for node_key, data in nodes.items():
    #             # ------ SCALAR mode -------
    #             if search_type not in ('hhi', 'hhg', 'hoi'):
    #                 # handle shorthand {'type':'action','action':'sitting'}
    #                 filters = dict(attr_dict)
    #                 t = filters.pop('type', None)
    #                 if t and t not in ('human', 'object'):
    #                     # means t names the field to filter on
    #                     filters = {t: filters.get(t)}
    #                 elif t:
    #                     filters['type'] = t
    #                 # now drop 'type'
    #                 filters.pop('type', None)
    #
    #                 # node‐type filter
    #                 if 'type' in filters and data.get('type') != filters['type']:
    #                     continue
    #                 # other scalar filters
    #                 if any(data.get(k) != v for k, v in filters.items()):
    #                     continue
    #
    #                 # emit one record
    #                 rec = {'frame': frame_num}
    #                 for f in return_fields:
    #                     rec[f] = node_key if f == 'id' else data.get(f)
    #                 results.append(rec)
    #
    #             # ------ INTERACTION mode -------
    #             else:
    #                 # only humans can have hhi/hhg/hoi edges
    #                 if data.get('type') != 'human':
    #                     continue
    #
    #                 is_human_inter = search_type in ('hhi', 'hhg')
    #                 desired_edge = 'geometric' if search_type == 'hhg' else 'physical'
    #
    #                 # scan exactly the same edges as extract_entity_time_slots
    #                 for _, tgt, _, e in self.graph.out_edges(node_key, data=True, keys=True):
    #                     if e.get('edge_type') != desired_edge:
    #                         continue
    #
    #                     tgt_data = self.frame_data[frame_id].get(tgt, {})
    #                     tgt_type = tgt_data.get('type')
    #
    #                     # filter target type
    #                     if is_human_inter and tgt_type != 'human':
    #                         continue
    #                     if not is_human_inter and search_type == 'hoi' and tgt_type != 'object':
    #                         continue
    #
    #                     # build the event dict
    #                     ev = {
    #                         'frame': frame_num,
    #                         'id': node_key,
    #                         'interactions': e.get('relation_type'),
    #                         'id_pair': tgt,
    #                         'type': search_type,
    #                         'bbox': data.get('bbox')
    #                     }
    #                     if is_human_inter:
    #                         ev.update({
    #                             'gender_pair': tgt_data.get('gender'),
    #                             'age_pair': tgt_data.get('age'),
    #                             'race_pair': tgt_data.get('race'),
    #                            'bbox_pair': tgt_data.get('bbox')
    #                         })
    #                     else:  # hoi
    #                         ev.update({
    #                             'obj_name': tgt_data.get('obj_name'),
    #                             'tracking_id': tgt_data.get('tracking_id'),
    #                             'category_id': tgt_data.get('category_id'),
    #                         })
    #
    #                     # apply any extra_filters (e.g. interactions, gender_pair, etc.)
    #                     if any(ev.get(k) != v for k, v in extra_filters.items()):
    #                         continue
    #
    #                     # output only requested fields
    #                     rec = {}
    #                     for f in return_fields:
    #                         rec[f] = ev.get(f)
    #                     results.append(rec)
    #
    #     return results
    # def _search_graph(self, attr_dict, return_fields, time_window):
    #     """
    #     Search scalar or interaction events in a frame window.
    #
    #     :param attr_dict: filters. For interactions include 'type':'hhi'|'hhg'|'hoi'
    #                       plus any of 'interactions','gender_pair','age_pair','race_pair'
    #                       (for hhi/hhg) or 'obj_name','tracking_id','category_id' (for hoi).
    #     :param return_fields: list of keys to return, e.g. ['frame','id','interactions',...].
    #     :param time_window: (start_frame, end_frame) inclusive.
    #     """
    #     # validate inputs
    #     if not isinstance(attr_dict, dict) \
    #             or not isinstance(return_fields, list) \
    #             or not isinstance(time_window, tuple):
    #         raise ValueError("Invalid attr_dict, return_fields, or time_window")
    #
    #     attr_dict = {'age': 'middle_aged', 'type': 'age'}
    #     start_frame, end_frame = time_window
    #     search_type = attr_dict.get('type')
    #
    #     # --- SPECIAL CASE: return only id and bbox for human nodes ---
    #     if search_type == 'human' and len(attr_dict) == 1:
    #         results = []
    #         for frame_id, nodes in self.frame_data.items():
    #             try:
    #                 frame_num = int(frame_id.split('.')[0])
    #             except (ValueError, IndexError):
    #                 continue
    #             if not (start_frame <= frame_num <= end_frame):
    #                 continue
    #
    #             for node_key, data in nodes.items():
    #                 if data.get('type') != 'human':
    #                     continue
    #                 rec = {
    #                     'frame': frame_num,
    #                     'id': node_key,
    #                     'bbox': data.get('bbox')
    #                 }
    #                 results.append(rec)
    #         return results
    #     # ---------------------------------------------------------------
    #
    #     extra_filters = {k: v for k, v in attr_dict.items() if k != 'type'}
    #     results = []
    #
    #     for frame_id, nodes in self.frame_data.items():
    #         # parse frame number
    #         try:
    #             frame_num = int(frame_id.split('.')[0])
    #         except (ValueError, IndexError):
    #             continue
    #         if not (start_frame <= frame_num <= end_frame):
    #             continue
    #
    #         for node_key, data in nodes.items():
    #             if data.get('face_visibility') == 'invisible':
    #                continue
    #             # ------ SCALAR mode -------
    #             if search_type not in ('hhi', 'hhg', 'hoi'):
    #                 # handle shorthand {'type':'action','action':'sitting'}
    #                 filters = dict(attr_dict)
    #                 t = filters.pop('type', None)
    #                 if t and t not in ('human', 'object'):
    #                     filters = {t: filters.get(t)}
    #                 elif t:
    #                     filters['type'] = t
    #                 filters.pop('type', None)
    #
    #                 # node-type filter
    #                 if 'type' in filters and data.get('type') != filters['type']:
    #                     continue
    #                 # other scalar filters
    #                 if any(data.get(k) != v for k, v in filters.items()):
    #                     continue
    #
    #                 # emit one record
    #                 rec = {'frame': frame_num}
    #                 for f in return_fields:
    #
    #                     rec[f] = node_key if f == 'id' else data.get(f)
    #                 results.append(rec)
    #
    #             # ------ INTERACTION mode -------
    #             else:
    #                 # only humans can have hhi/hhg/hoi edges
    #                 if data.get('type') != 'human':
    #                     continue
    #
    #                 is_human_inter = search_type in ('hhi', 'hhg')
    #                 desired_edge = 'geometric' if search_type == 'hhg' else 'physical'
    #
    #                 for _, tgt, _, e in self.graph.out_edges(node_key, data=True, keys=True):
    #                     if e.get('edge_type') != desired_edge:
    #                         continue
    #
    #                     tgt_data = self.frame_data[frame_id].get(tgt, {})   # node information
    #                     tgt_type = tgt_data.get('type')
    #
    #                     # filter target type
    #                     if is_human_inter and tgt_type != 'human':
    #                         continue
    #                     if not is_human_inter and search_type == 'hoi' and tgt_type != 'object':
    #                         continue
    #
    #                     # build the event dict
    #                     ev = {
    #                         'frame': frame_num,
    #                         'id': node_key,
    #                         'interactions': e.get('relation_type'),
    #                         'id_pair': tgt,
    #                         'type': search_type,
    #                         'bbox': data.get('bbox')
    #                     }
    #                     if is_human_inter:
    #                         ev.update({
    #                             'gender_pair': tgt_data.get('gender'),
    #                             'age_pair': tgt_data.get('age'),
    #                             'race_pair': tgt_data.get('race'),
    #                             'bbox_pair': tgt_data.get('bbox')
    #                         })
    #                     else:  # hoi
    #                         ev.update({
    #                             'obj_name': tgt_data.get('obj_name'),
    #                             'tracking_id': tgt_data.get('tracking_id'),
    #                             'category_id': tgt_data.get('category_id'),
    #                         })
    #
    #                     # apply any extra_filters
    #                     if any(ev.get(k) != v for k, v in extra_filters.items()):
    #                         continue
    #
    #                     # output only requested fields
    #                     rec = {f: ev.get(f) for f in return_fields}
    #                     if rec not in results:
    #                        results.append(rec)
    #
    #     return results
    def _search_graph(self, attr_dict, return_fields, time_window):
        """
        Search the scene graph for nodes matching attributes within a frame window.

        Behavior:
          1) Restrict to frames in [start_frame, end_frame]
          2) Within each frame, restrict nodes by attr_dict['node_type'] if provided
          3) Apply ONLY filters listed in attr_dict['asked_attributes'].
             - Scalar filters: e.g. {'age':'middle-aged'}
             - Interaction filters: keys 'hhi'|'hhg'|'hoi' with dict of pair/edge constraints.
               * hhi -> edge_type='physical', target must be human
               * hhg -> edge_type='geometric', target must be human
               * hoi -> edge_type='physical', target must be object

        Returns a list of dicts; always includes 'frame'. Other fields follow `return_fields`:
          id -> source node_key
          bbox -> source bbox
          id_pair -> target node_key (for interactions)
          bbox_pair -> target bbox (for interactions)
          interactions -> edge relation_type (if requested)
        """
        # --- validate inputs ---
        if not isinstance(attr_dict, dict) or not isinstance(return_fields, list) or not isinstance(time_window, tuple):
            raise ValueError("Invalid arguments")
        if len(time_window) != 2:
            raise ValueError("time_window must be a (start_frame, end_frame) tuple")

        start_frame, end_frame = time_window
        node_type = attr_dict.get('node_type')  # e.g., 'human' or 'object'
        asked = attr_dict.get('asked_attributes', {}) or {}
        if not isinstance(asked, dict):
            raise ValueError("'asked_attributes' must be a dict when provided")

        # Split scalar vs interaction filters
        INTER_KEYS = {'hhi', 'hhg', 'hoi'}
        scalar_filters = {k: v for k, v in asked.items() if k not in INTER_KEYS}
        interaction_filters = {k: v for k, v in asked.items() if k in INTER_KEYS}

        def _matches(val, expected):
            """Membership if expected is a collection, exact match otherwise."""
            if isinstance(expected, (list, tuple, set)):
                return val in expected
            return val == expected

        def _emit_record(frame_num, src_key, src_data, tgt_key=None, tgt_data=None, edge_rel=None):
            """Build a record based on requested return_fields."""
            rec = {'frame': frame_num}
            for f in return_fields:
                if f == 'id':
                    rec['id'] = src_key
                elif f == 'bbox':
                    rec['bbox'] = src_data.get('bbox')
                elif f == 'id_pair':
                    rec['id_pair'] = tgt_key
                elif f == 'bbox_pair':
                    rec['bbox_pair'] = (tgt_data or {}).get('bbox')
                elif f == 'interactions':
                    rec['interactions'] = edge_rel
                else:
                    # Try source first, then target as a fallback
                    rec[f] = src_data.get(f, (tgt_data or {}).get(f))
            return rec

        results = []

        # --- scan frames in window ---
        for frame_id, nodes in self.frame_data.items():
            try:
                frame_num = int(frame_id.split('.')[0])
            except (ValueError, IndexError):
                continue
            if not (start_frame <= frame_num <= end_frame):
                continue

            # --- per-node within frame ---
            for node_key, data in nodes.items():
                # 1) node_type filter (if provided)
                if node_type and data.get('type') != node_type:
                    continue

                # 2) scalar filters on the *source* node
                ok = True
                for k, v in scalar_filters.items():
                    if not _matches(data.get(k), v):
                        ok = False
                        break
                if not ok:
                    continue

                # 3) interaction filters (each key acts as an additional constraint)
                if interaction_filters:
                    # For each interaction type requested, we require at least one matching edge
                    for inter_key, inter_req in interaction_filters.items():
                        if not isinstance(inter_req, dict):
                            inter_req = {}

                        # determine edge/target expectations
                        if inter_key == 'hhi':
                            desired_edge = 'physical'
                            desired_tgt_type = 'human'
                        elif inter_key == 'hhg':
                            desired_edge = 'geometric'
                            desired_tgt_type = 'human'
                        elif inter_key == 'hoi':
                            desired_edge = 'physical'
                            desired_tgt_type = 'object'
                        else:
                            # unknown interaction key — skip
                            continue

                        found_any = False
                        # Examine outgoing edges from THIS node_key in the global graph
                        for _, tgt, _, e in self.graph.out_edges(node_key, data=True, keys=True):
                            if e.get('edge_type') != desired_edge:
                                continue

                            # Get target node's per-frame attributes (prefer frame_data for correctness per frame)
                            tgt_data = nodes.get(tgt, {})  # nodes == self.frame_data[frame_id]
                            tgt_type = tgt_data.get('type') or self.graph.nodes.get(tgt, {}).get('node_type')
                            if tgt_type != desired_tgt_type:
                                continue

                            # Assemble the “pair” view for filtering
                            edge_rel = e.get('relation_type')
                            pair_view = {
                                'interactions': edge_rel,
                                'gender_pair': tgt_data.get('gender'),
                                'age_pair': tgt_data.get('age'),
                                'race_pair': tgt_data.get('race'),
                                'obj_name': tgt_data.get('obj_name'),
                                'tracking_id': tgt_data.get('tracking_id'),
                                'category_id': tgt_data.get('category_id'),
                            }

                            # Apply interaction-specific filters (only those keys present in inter_req)
                            inter_ok = True
                            for key, expected in inter_req.items():
                                if not _matches(pair_view.get(key), expected):
                                    inter_ok = False
                                    break
                            if not inter_ok:
                                continue

                            # Passed: emit a record for this edge
                            results.append(_emit_record(
                                frame_num=frame_num,
                                src_key=node_key,
                                src_data=data,
                                tgt_key=tgt,
                                tgt_data=tgt_data,
                                edge_rel=edge_rel
                            ))
                            found_any = True

                        # If this interaction filter was present but no edge matched, the source node fails overall
                        if not found_any:
                            ok = False
                            break

                    if not ok:
                        continue

                else:
                    # No interaction filters; emit a single record for the node itself
                    results.append(_emit_record(
                        frame_num=frame_num,
                        src_key=node_key,
                        src_data=data
                    ))

        # stable ordering
        results.sort(key=lambda r: (r['frame'], r.get('id', ''), r.get('id_pair', '')))

        def _freeze(o):
            """Recursively convert containers to hashable forms for set membership."""
            if isinstance(o, dict):
                return tuple(sorted((k, _freeze(v)) for k, v in o.items()))
            elif isinstance(o, (list, tuple)):
                return tuple(_freeze(x) for x in o)
            elif isinstance(o, set):
                return tuple(sorted(_freeze(x) for x in o))
            # Optional: handle numpy scalars/arrays if you have them
            try:
                import numpy as np
                if isinstance(o, np.ndarray):
                    return ('ndarray', tuple(o.tolist()))
                if isinstance(o, (np.integer, np.floating, np.bool_)):
                    return o.item()
            except Exception:
                pass
            return o

        unique, seen = [], set()
        for rec in results:
            fp = _freeze(rec)
            if fp not in seen:
                unique.append(rec)
                seen.add(fp)
        results = unique

        return results

    def visualize_frame(self, frame_id):
        """
        Visualize graph nodes + edges for a single frame.

        Args:
            frame_id (str): مثلا "000885.jpg"
        """

        # 1) چک کن فریم وجود دارد یا نه
        if frame_id not in self.frame_data:
            print(f"❌ Frame {frame_id} not found!")
            return

        # 2) نودهای مربوط به این فریم را بگیر
        frame_nodes = list(self.frame_data[frame_id].keys())

        # 3) یک Subgraph بساز فقط با نودهای همان فریم
        subG = self.graph.subgraph(frame_nodes).copy()

        print(f"✅ Visualizing Frame: {frame_id}")
        print(f"   Nodes: {len(subG.nodes)}")
        print(f"   Edges: {len(subG.edges)}")

        # 4) Layout
        pos = nx.spring_layout(subG, seed=42)

        # 5) رنگ نودها بر اساس نوع
        node_colors = []
        for n in subG.nodes:
            n_type = subG.nodes[n].get("node_type", "unknown")
            if n_type == "human":
                node_colors.append("skyblue")
            else:
                node_colors.append("lightgreen")

        # 6) رسم گراف
        plt.figure(figsize=(10, 8))

        nx.draw(
            subG,
            pos,
            with_labels=True,
            node_color=node_colors,
            node_size=1500,
            font_size=8,
            edge_color="gray"
        )

        # 7) Edge labels (relation_type)
        edge_labels = {
            (u, v): d.get("relation_type", "")
            for u, v, d in subG.edges(data=True)
        }

        nx.draw_networkx_edge_labels(
            subG,
            pos,
            edge_labels=edge_labels,
            font_size=7
        )

        plt.title(f"Scene Graph Visualization - Frame {frame_id}")
        plt.show()



















