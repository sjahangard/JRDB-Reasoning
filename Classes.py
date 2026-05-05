# import copy
#
# class DataDictionary:
#     def __init__(self, task_type="", type_="", difficulty_level=0, question_level="",
#                  seq="", modality_type="", frame_rate=None, spatial_level=0, temporal_level=0,
#                  question="", entities=None, timestamps=None, labels=None):
#         # Task information
#         self.task_type = task_type     # e.g., "VG" or "VQA"
#         self.type = type_              # e.g., "human"
#         self.difficulty_level = difficulty_level  # Difficulty level (numeric)
#         self.question_level = question_level      # e.g., "U(S)U(T)"
#
#         # Data retrieval information
#         self.data = {
#             "seq": seq,                          # Sequence name
#             "modality_type": modality_type,      # e.g., "image"
#             "specs": {
#                 "fps": frame_rate                # Frame rate in fps
#             }
#         }
#
#         # Question details
#         self.question = {
#             "question": question if question else {},                # e.g., "Find a young person..."
#             "spatial_level": spatial_level,      # Spatial level (e.g., 1)
#             "temporal_level": temporal_level,    # Temporal level (e.g., 1)
#             "entities": entities if entities else {},  # Entity attributes
#             "timestamps": timestamps if timestamps else {}  # Timestamp information
#         }
#
#         # Labels
#         self.labels = labels if labels else {}     # e.g., bounding boxes, IDs
#
#     def __getitem__(self, key):
#         return getattr(self, key)
#
#     def __setitem__(self, key, value):
#         setattr(self, key, value)
#
#     def copy(self):
#         # Create a deep copy of the current instance
#         return copy.deepcopy(self)
#
#     def __repr__(self):
#         return (f"DataDictionary(task_type={self.task_type}, type={self.type}, "
#                 f"difficulty_level={self.difficulty_level}, question_level={self.question_level}, "
#                 f"data={self.data}, question={self.question}, labels={self.labels})")
import copy


class DataDictionary:
    def __init__(self, task_type="", type_="", difficulty_level=0, question_level="",
                 seq="", modality_type="", frame_rate=None, spatial_level=0, temporal_level=0,
                 question="", entities=None, timestamps=None, labels=None):
        # Task information
        self.task_type = task_type
        self.type = type_
        self.difficulty_level = difficulty_level
        self.question_level = question_level

        # Data retrieval information
        self.data = {
            "seq": seq,
            "modality_type": modality_type,
            "specs": {
                "fps": frame_rate
            }
        }

        # Question details
        self.question = {
            "question": question if question else {},
            "spatial_level": spatial_level,
            "temporal_level": temporal_level,
            "entities": entities if entities else {},
            "timestamps": timestamps if timestamps else {}
        }

        # Labels
        self.labels = labels if labels else {}

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __deepcopy__(self, memo):
        # Create a shallow copy first, then deep copy nested attributes
        new_obj = copy.copy(self)  # shallow copy

        # Deep copy each mutable attribute
        new_obj.data = copy.deepcopy(self.data, memo)
        new_obj.question = copy.deepcopy(self.question, memo)
        new_obj.labels = copy.deepcopy(self.labels, memo)

        return new_obj

    def copy(self):
        return copy.deepcopy(self)

    def __repr__(self):
        return (f"DataDictionary(task_type={self.task_type}, type={self.type}, "
                f"difficulty_level={self.difficulty_level}, question_level={self.question_level}, "
                f"data={self.data}, question={self.question}, labels={self.labels})")
