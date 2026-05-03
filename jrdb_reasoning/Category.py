
train_seq = [
    'bytes-cafe-2019-02-07_0.json',
    # 'clark-center-2019-02-28_1.json',
    # 'nvidia-aud-2019-04-18_0.json',
    # 'clark-center-2019-02-28_0.json',
    # 'clark-center-intersection-2019-02-28_0.json',
    # 'cubberly-auditorium-2019-04-22_0.json',
    # 'forbes-cafe-2019-01-22_0.json',
    # 'gates-159-group-meeting-2019-04-03_0.json',
    # 'gates-basement-elevators-2019-01-17_1.json',
    # 'gates-to-clark-2019-02-28_1.json',
    # 'hewlett-packard-intersection-2019-01-24_0.json',
    # 'huang-basement-2019-01-25_0.json',
    # 'huang-lane-2019-02-12_0.json',
    # 'jordan-hall-2019-04-22_0.json',
    # 'memorial-court-2019-03-16_0.json',
    # 'packard-poster-session-2019-03-20_0.json', 'packard-poster-session-2019-03-20_1.json',
    # 'packard-poster-session-2019-03-20_2.json',
    # 'stlc-111-2019-04-19_0.json',
    # 'svl-meeting-gates-2-2019-04-08_0.json',
    # 'svl-meeting-gates-2-2019-04-08_1.json',
    # 'tressider-2019-03-16_0.json',
    # 'clark-center-2019-02-28_1.json',
    # 'gates-ai-lab-2019-02-08_0.json',
    # 'huang-2-2019-01-25_0.json',
    # 'meyer-green-2019-03-16_0.json', 'nvidia-aud-2019-04-18_0.json',
    # 'tressider-2019-03-16_1.json',
    # 'tressider-2019-04-26_2.json'
]
test_seq = [
       'cubberly-auditorium-2019-04-22_1.json',
       'discovery-walk-2019-02-28_0.json', 'discovery-walk-2019-02-28_1.json', 'food-trucks-2019-02-12_0.json',
    'gates-ai-lab-2019-04-17_0.json', 'gates-basement-elevators-2019-01-17_0.json'
    # , 'gates-foyer-2019-01-17_0.json',
    # 'gates-to-clark-2019-02-28_0.json',
    # 'hewlett-class-2019-01-23_0.json',
    # 'hewlett-class-2019-01-23_1.json', 'huang-2-2019-01-25_1.json', 'huang-intersection-2019-01-22_0.json',
    # 'indoor-coupa-cafe-2019-02-06_0.json', 'lomita-serra-intersection-2019-01-30_0.json', 'meyer-green-2019-03-16_1.json', 'nvidia-aud-2019-01-25_0.json',
    # 'nvidia-aud-2019-04-18_1.json',
    # 'nvidia-aud-2019-04-18_2.json', 'outdoor-coupa-cafe-2019-02-06_0.json', 'quarry-road-2019-02-28_0.json',
    #
    # 'serra-street-2019-01-30_0.json', 'stlc-111-2019-04-19_1.json', 'stlc-111-2019-04-19_2.json', 'tressider-2019-03-16_2.json', 'tressider-2019-04-26_0.json',
    # 'tressider-2019-04-26_1.json', 'tressider-2019-04-26_3.json'
     ]


GENDERS = ['Male', 'Female']
AGES = ['Childhood', 'Adolescence', 'Young_Adulthood', 'Middle_Adulthood', 'Late_Adulthood']
RACES = ['Caucasian/White', 'Negroid/Black', 'Mongoloid/Asian']
ACTIONS = ['walking', 'standing', 'holding sth', 'sitting', 'listening to someone', 'talking to someone',
           'looking at robot', 'looking into sth', 'looking at sth', 'cycling', 'going upstairs', 'bending',
           'typing', 'interaction with door', 'eating sth', 'talking on the phone', 'pointing at sth',
           'going downstairs', 'reading', 'pushing', 'skating', 'scootering', 'greeting gestures', 'running',
           'writing', 'pulling', 'lying']



class_colors = {
        'door': (0, 0, 200),  # Red
        'road': (128, 128, 128),  # Middle Gray
        'machines': (200, 0, 0),  # Blue
        'child': (0, 200, 200),  # Yellow
        'glass': (200, 0, 200),  # Magenta
        'trash_bin': (200, 200, 0),  # Cyan
        'shelf': (0, 0, 100),  # Dark Red
        'terrain': (0, 100, 0),  # Dark Green
        'car': (100, 0, 0),  # Dark Blue
        'bicycle/scooter': (0, 100, 100),  # Dark Yellow
        'bag': (100, 0, 100),  # Dark Magenta
        'big_vehicle': (100, 100, 0),  # Dark Cyan
        'chair/sofa': (0, 75, 150),  # Brown
        'pedestrian': (180, 105, 255),  # Hot Pink
        'vegetation': (0, 255, 0),  # Lime
        'sign': (128, 0, 128),  # Purple
        'trolley': (128, 128, 128),  # Grey
        'pole/trafficcone': (0, 165, 255),  # Orange
        'wall': (128, 128, 0),  # Teal
        # 'misc': (0, 0, 128),  # Maroon
        'column': (128, 0, 0),  # Navy
        'bicyclist/rider': (212, 255, 127),  # Aquamarine
        'sky': (255, 191, 0),  # Deep Sky Blue
        'stair': (30, 105, 210),  # Chocolate
        'window': (147, 20, 255),  # Deep Pink
        'building': (240, 255, 255),  # Ivory
        'ceiling': (140, 230, 240),  # Khaki
        'table': (196, 228, 255),  # Bisque
        'barrier/fence': (170, 178, 32),  # Light Sea Green
        'floor/walking_path': (209, 206, 0),  # Dark Turquoise
        'skateboard/segway/hoverboard': (42, 42, 165),  # Brown
        'crutch': (235, 206, 135),  # Tan
        'fire_extinguisher': (255, 69, 0),  # Red-Orange
        'big_socket/button': (60, 179, 113),  # Medium Sea Green
        'airpot': (72, 61, 139),  # DarkSlateBlue
        'controller': (100, 149, 237),  # Cornflower Blue
        'umbrella': (255, 105, 180),  # HotPink
        # 'big button/socket': (85, 107, 47),  # Dark Olive Green
        'animal': (210, 105, 30),  # Chocolate
        'monitor': (220, 220, 220),  # Gainsboro
        'carpet': (105, 105, 105),  # Dim Gray
        'cabinet': (107, 142, 35),  # Olive Drab
        'jacket': (123, 104, 238),  # MediumSlateBlue
        'emergency_pole': (178, 34, 34),  # Firebrick
        'helmet': (154, 205, 50),  # YellowGreen
        'peripheral': (240, 128, 128),  # Light Coral
        'curtain': (95, 158, 160),  # Cadet Blue
        'box': (188, 143, 143),  # Rosy Brown
        'Board': (0, 255, 127),  # Spring Green
        'document': (70, 130, 180),  # Steel Blue
        'television': (240, 230, 140),  # Khaki
        'light_pole': (255, 127, 80),  # Coral
        'board': (119, 136, 153),  # Light Slate Gray
        'manhole': (176, 224, 230),  # Powder Blue
        'phone': (255, 218, 185),  # Peach Puff
        'waterbottle': (255, 160, 122),  # Light Salmon
        'picture_frames': (175, 238, 238),  # PaleTurquoise
        'standee': (205, 133, 63),  # Peru
        'store_sign': (250, 128, 114),  # Salmon
        'statues': (233, 150, 122),  # DarkSalmon
        'cargo': (173, 216, 230),  # LightBlue
        'tableware': (255, 255, 0),  # Cyan
        'accessory': (250, 250, 210),  # LightGoldenrodYellow
        'other': (192, 192, 192),  # Silver
        'clock': (135, 206, 235),  # SkyBlue
        'golfcart': (216, 191, 216),  # Thistle
        'decoration': (221, 160, 221),  # Plum
        'hanging_light': (218, 112, 214),  # Orchid
        'wall_panel': (34, 139, 34),  # ForestGreen
        'ladder': (219, 112, 147),  # PaleVioletRed
        'poster': (0, 215, 255),  # Gold
        'lift': (147, 112, 219),  # Slate Gray
        'vent': (0, 128, 128),  # Teal
        'fountain': (139, 0, 139),  # DarkMagenta
        'door_handle': (0, 139, 139),  # DarkCyan
    }
BODY_CONTENTS_inter = {
    "floor":'walking',
    "ground":'walking',
    "chair":'sitting',
    "sidewalk":'walking',
    "bike":'riding',
    "stairs":'going up',
    "platform":'walking',
    "sofa":'sitting',
    "grass":'walking',
    "street":'walking',
    "crosswalk":'walking',
    "road":'walking',
    "scooter":'scootering',
    "skateboard":'riding',
    "pathway":'walking',
    "desk":'leaning',
    "balcony":'standing',
    "bench":'sitting'
}
HUMAN_HUMAN_INTERACTION = [
    'bending together', 'conversation', 'cycling together', 'eating together', 'going downstairs together',
    'going upstairs together', 'holding something together', 'holding sth together', 'hugging',
    'interaction with door together', 'looking at robot together', 'looking at something together',
    'looking at sth together', 'looking into sth together', 'moving together', 'shaking hand', 
    'sitting together', 'standing together', 'walking together', 'walking toward each other',
    'waving hand together', 'pointing at sth together'
]

Posed_HUMAN_HUMAN_INTERACTION = [
    'bending together', 'cycling together', 'going downstairs together',
    'going upstairs together', 'moving together',
    'sitting together', 'standing together', 'walking together', 'walking toward each other'
]

GEOMETRY = [
    "left",
    "right",
    "front",
    "back",
    'front left',
    'back left',
    'front right',
    'back right',

    "up left",
    "up right",
    "up front",
    "up back",
    'up front left',
    'up back left',
    'up front right',
    'up back right',

    "down left",
    "down right",
    "down front",
    "down back",
    'down front left',
    'down back left',
    'down front right',
    'down back right',
]

HUMAN_OBJECT_INTERACTION = [
    'working', 'opening', 'closing', 'entering', 'exiting', 'touching', 'cleaning', 'watering', 'pruning', 'planting',
    'sitting', 'pushing', 'pulling', 'carrying', 'holding', 'looking at', 'presenting', 'organizing', 'leaning',
    'painting', 'installing', 'removing', 'walking', 'standing', 'running', 'repairing', 'riding', 'parking',
    'mounting', 'depositing', 'placing', 'warning', 'blocking', 'adjusting', 'constructing', 'maintaining',
    'handling/manipulating', 'accessing', 'inspecting', 'storing', 'looking out', 'pressing', 'switching', 'drinking',
    'refilling', 'moving', 'operating', 'displaying', 'arranging', 'talking on', 'browsing/engaging', 'loading',
    'unloading', 'transporting', 'connecting', 'using', 'configuring', 'disconnecting', 'printing/copying',
    'identifying', 'locating', 'promoting', 'advertising', 'finding_something', 'putting_something', 'directing',
    'attracting', 'investigating', 'exploring', 'utilizing', 'understanding', 'observing', 'turning', 'wearing',
    'adorning', 'replacing', 'matching', 'collecting', 'folding', 'servicing', 'feeding', 'climbing', 'going down',
    'going up', 'looking into'
]


None_posed_HUMAN_OBJECT_INTERACTION = [
    'working', 'opening', 'closing', 'entering', 'exiting', 'touching', 'cleaning', 'watering', 'pruning', 'planting',
    'pushing', 'pulling', 'carrying', 'holding', 'looking at', 'presenting', 'organizing', 'leaning',
    'painting', 'installing', 'removing', 'repairing', 'riding', 'parking',
    'mounting', 'depositing', 'placing', 'warning', 'blocking', 'adjusting', 'constructing', 'maintaining',
    'handling/manipulating', 'accessing', 'inspecting', 'storing', 'looking out', 'pressing', 'switching', 'drinking',
    'refilling', 'operating', 'displaying', 'arranging', 'talking on', 'browsing/engaging', 'loading',
    'unloading', 'transporting', 'connecting', 'using', 'configuring', 'disconnecting', 'printing/copying',
    'identifying', 'locating', 'promoting', 'advertising', 'finding_something', 'putting_something', 'directing',
    'attracting', 'investigating', 'exploring', 'utilizing', 'understanding', 'observing', 'turning', 'wearing',
    'adorning', 'replacing', 'matching', 'collecting', 'folding', 'servicing', 'feeding', 'climbing', 'looking into']

SOCIAL_AIMS = [
    "eating/ordering_food", "studying/writing/reading/working", "socializing", "commuting", "unknown", "navigating",
    "waiting_for_someone/something", "wandering", "discussing_an_object/matter", "excursion", "attending_class/lecture/seminar"
]

SALIENT_OBJECTS = [
    "gate", "table", "counter", "door", "pillar", "shelves", "wall", "standboard", "poster", "desk", "food-truck",
    "bike", "chair", "stairs", "fence", "show-case", "room", "board", "cabinet", "garbage-bin", "stroller", "elevator",
    "buffet-cafeteria", "trolley", "forecourt", "scooter", "bus", "robot", "platform", "window", "tree", "pole",
    "crutches", "stand-pillar", "screen", "car", "copy-machine", "class", "coffee-machine", "balcony", "sofa",
    "statue", "floor", "bench", "building", "baggage", "shop", "light-street", "drink-fountain"
]

BODY_CONTENTS = [
    "floor", "ground", "chair", "sidewalk", "bike", "stairs", "platform", "sofa", "grass", "street", "crosswalk",
    "road", "scooter", "skateboard", "pathway", "desk", "balcony", "bench"
]

OBJECTS = [
    # "pedestrian",
    "chair/sofa",
    "bag",
    "table",
    "machines",
    "door",
    "hanging_light",
    # "glass",
    # "vegetation",
    "board",
    "sign",
    "wall",
    "barrier/fence",
    # "floor/walking_path",
    # "ceiling",
    "bicycle/scooter",
    "light_pole",
    "trash_bin",
    "pole/trafficcone",
    # "building",
    "tableware",
    # "column",
    "cabinet",
    "shelf",
    "window",
    "terrain",
    "stair",
    "big_socket/button",
    "television",
    # "sky",
    "waterbottle",
    "picture_frames",
    "box",
    "car",
    "monitor",
    # "bicyclist/rider",
    "controller",
    "manhole",
    "jacket",
    "decoration",
    "standee",
    # "road",
    "phone",
    "poster",
    "store_sign",
    "carpet",
    "trolley",
    # "other",
    "document",
    # "child",
    "airpot",
    "cargo",
    # "peripheral",
    "vent",
    "skateboard/segway/hoverboard",
    "clock",
    "door_handle",
    "statues",
    "helmet",
    "curtain",
    "golfcart",
    "fountain",
    "wall_panel",
    "emergency_pole",
    "accessory",
    "crutch",
    "big_vehicle",
    "fire_extinguisher",
    "umbrella",
    "lift",
    "animal",
    "ladder"
]

# all_human_pose_action = ['walking', 'standing', 'sitting', 'cycling', 'going upstairs', 'bending', 'going downstairs', 'skating', 'scootering', 'running']
all_human_pose_action = ['walking', 'standing', 'sitting', 'bending', 'running','cycling', 'going upstairs', 'bending', 'going downstairs','skating', 'scootering']
all_human_pose_action_postfix = ['walking', 'standing', 'sitting','running','going up','going down']
all_human_pose_action_postfix_hoi = ['walking', 'standing']
Indoor_seq=[]
Outdoor_seq=[]

train_seqs = [
    # 'bytes-cafe-2019-02-07_0', 'clark-center-2019-02-28_0',
                               'clark-center-intersection-2019-02-28_0',
                               'cubberly-auditorium-2019-04-22_0', 'forbes-cafe-2019-01-22_0',
                               'gates-159-group-meeting-2019-04-03_0',
                               'gates-basement-elevators-2019-01-17_1', 'gates-to-clark-2019-02-28_1',
                               'hewlett-packard-intersection-2019-01-24_0',
                               'huang-basement-2019-01-25_0', 'huang-lane-2019-02-12_0', 'jordan-hall-2019-04-22_0',
                               'memorial-court-2019-03-16_0',
                               'packard-poster-session-2019-03-20_0', 'packard-poster-session-2019-03-20_1',
                               'packard-poster-session-2019-03-20_2',
                               'stlc-111-2019-04-19_0', 'svl-meeting-gates-2-2019-04-08_0',
                               'svl-meeting-gates-2-2019-04-08_1', 'tressider-2019-03-16_0']

valid_seqs = ['clark-center-2019-02-28_1', 'gates-ai-lab-2019-02-08_0', 'huang-2-2019-01-25_0',
                               'meyer-green-2019-03-16_0', 'nvidia-aud-2019-04-18_0',
                               'tressider-2019-03-16_1', 'tressider-2019-04-26_2']

test_seqs = ['cubberly-auditorium-2019-04-22_1', 'discovery-walk-2019-02-28_0', 'discovery-walk-2019-02-28_1', 'food-trucks-2019-02-12_0',
                              'gates-ai-lab-2019-04-17_0', 'gates-basement-elevators-2019-01-17_0', 'gates-foyer-2019-01-17_0', 'gates-to-clark-2019-02-28_0',
                              'hewlett-class-2019-01-23_0', 'hewlett-class-2019-01-23_1', 'huang-2-2019-01-25_1', 'huang-intersection-2019-01-22_0',
                              'indoor-coupa-cafe-2019-02-06_0', 'lomita-serra-intersection-2019-01-30_0', 'meyer-green-2019-03-16_1', 'nvidia-aud-2019-01-25_0',
                              'nvidia-aud-2019-04-18_1', 'nvidia-aud-2019-04-18_2', 'outdoor-coupa-cafe-2019-02-06_0', 'quarry-road-2019-02-28_0',
                              'serra-street-2019-01-30_0', 'stlc-111-2019-04-19_1', 'stlc-111-2019-04-19_2', 'tressider-2019-03-16_2', 'tressider-2019-04-26_0',
                              'tressider-2019-04-26_1', 'tressider-2019-04-26_3']

all_class_id= {
    'glass': 0,
           'wall': 1,
           'door': 2,
           'road': 3,
           'terrain': 4, 'sky': 5,
           'vegetation': 6,
            'column': 7,
           'building': 8,
            'stair': 9,
           'ceiling': 10,
           'barrier/fence': 11, 'crutch': 12, 'fire_extinguisher': 13,
           'airpot': 14, 'controller': 15, 'umbrella': 16, 'animal': 17, 'carpet': 18, 'cabinet': 19, 'jacket': 20,
           'emergency_pole': 21, 'helmet': 22,
           'peripheral': 23,
           'curtain': 24, 'box': 25, 'document': 26,
           'television': 27, 'light_pole': 28, 'manhole': 29, 'phone': 30, 'ladder': 31, 'waterbottle': 32,
           'picture_frames': 33, 'standee': 34, 'store_sign': 35, 'statues': 36, 'cargo': 37, 'tableware': 38,
           'accessory': 39,
           'other': 40,
           'clock': 41, 'golfcart': 42, 'wall_panel': 43, 'hanging_light': 44,
           'window': 45, 'poster': 70, 'lift': 47, 'vent': 48, 'fountain': 49, 'door_handle': 50, 'monitor': 51,
           'board': 52,
    'floor/walking_path': 53,
    'machines': 54,
           'child': 55,
           'trash_bin': 56, 'shelf': 57, 'car': 58,
           'bicycle/scooter': 59, 'bag': 60, 'big_vehicle': 61, 'chair/sofa': 62,
           'pedestrian': 63,
           'trolley': 64,
           'pole/trafficcone': 65,
           'bicyclist/rider': 66,
           'table': 67, 'skateboard/segway/hoverboard': 68,
           'big_socket/button': 69, 'sign': 71, 'decoration': 72,
           'background': 73
           }
object_ids={0: 'road', 1: 'terrain', 2: 'sky', 3: 'vegetation', 4: 'wall', 5: 'column', 6: 'building', 7: 'stair', 8: 'ceiling', 9: 'barrier/fence', 10: 'crutch', 11: 'fire_extinguisher', 12: 'big_socket/button', 13: 'airpot', 14: 'controller', 15: 'umbrella', 16: 'animal', 17: 'monitor', 18: 'carpet', 19: 'cabinet', 20: 'jacket', 21: 'emergency_pole', 22: 'helmet', 23: 'peripheral', 24: 'curtain', 25: 'box', 26: 'document', 27: 'television', 28: 'light_pole', 29: 'board', 30: 'manhole', 31: 'phone', 32: 'ladder', 33: 'waterbottle', 34: 'picture_frames', 35: 'standee', 36: 'store_sign', 37: 'statues', 38: 'cargo', 39: 'tableware', 40: 'accessory', 41: 'other', 42: 'clock', 43: 'golfcart', 44: 'wall_panel', 45: 'decoration', 46: 'hanging_light', 47: 'window', 48: 'poster', 49: 'lift', 50: 'vent', 51: 'fountain', 52: 'door_handle', 53: 'floor/walking_path', 54: 'machines', 55: 'child', 56: 'trash_bin', 57: 'shelf', 58: 'car', 59: 'bicycle/scooter', 60: 'bag', 61: 'big_vehicle', 62: 'chair/sofa', 63: 'pedestrian', 64: 'sign', 65: 'trolley', 66: 'pole/trafficcone', 67: 'bicyclist/rider', 68: 'table', 69: 'skateboard/segway/hoverboard', 70: 'glass', 71: 'door'}
class_thing_id= {
    # 'glass': 0,
           # 'wall': 1,
           'door': 2,
           # 'road': 3,
           # 'terrain': 4, 'sky': 5,
           # 'vegetation': 6,
            'column': 7,
           'building': 8,
            'stair': 9,
           # 'ceiling': 10,
           'barrier/fence': 11, 'crutch': 12, 'fire_extinguisher': 13,
           'airpot': 14, 'controller': 15, 'umbrella': 16, 'animal': 17, 'carpet': 18, 'cabinet': 19, 'jacket': 20,
           'emergency_pole': 21, 'helmet': 22,
           # 'peripheral': 23,
           'curtain': 24, 'box': 25, 'document': 26,
           'television': 27, 'light_pole': 28, 'manhole': 29, 'phone': 30, 'ladder': 31, 'waterbottle': 32,
           'picture_frames': 33, 'standee': 34, 'store_sign': 35, 'statues': 36, 'cargo': 37, 'tableware': 38,
           'accessory': 39,
           # 'other': 40,
           'clock': 41, 'golfcart': 42, 'wall_panel': 43, 'hanging_light': 44,
           'window': 45, 'poster': 70, 'lift': 47, 'vent': 48, 'fountain': 49, 'door_handle': 50, 'monitor': 51,
           'board': 52,
    # 'floor/walking_path': 53,
    'machines': 54,
           # 'child': 55,
           'trash_bin': 56, 'shelf': 57, 'car': 58,
           'bicycle/scooter': 59, 'bag': 60, 'big_vehicle': 61, 'chair/sofa': 62,
           # 'pedestrian': 63,
           'trolley': 64,
           'pole/trafficcone': 65,
           # 'bicyclist/rider': 66,
           'table': 67, 'skateboard/segway/hoverboard': 68,
           'big_socket/button': 69, 'sign': 71, 'decoration': 72,
           # 'background': 73
           }

class_thing_human_id= {'glass': 0,
           # 'wall': 1,
           'door': 2,
           # 'road': 3,
           # 'terrain': 4, 'sky': 5,
           # 'vegetation': 6,
                       'column': 7,
           # 'building': 8,
                       'stair': 9,
           # 'ceiling': 10,
           'barrier/fence': 11, 'crutch': 12, 'fire_extinguisher': 13,
           'airpot': 14, 'controller': 15, 'umbrella': 16, 'animal': 17, 'carpet': 18, 'cabinet': 19, 'jacket': 20,
           'emergency_pole': 21, 'helmet': 22,
           # 'peripheral': 23,
           'curtain': 24, 'box': 25, 'document': 26,
           'television': 27, 'light_pole': 28, 'manhole': 29, 'phone': 30, 'ladder': 31, 'waterbottle': 32,
           'picture_frames': 33, 'standee': 34, 'store_sign': 35, 'statues': 36, 'cargo': 37, 'tableware': 38,
           'accessory': 39,
           # 'other': 40,
           'clock': 41, 'golfcart': 42, 'wall_panel': 43, 'hanging_light': 44,
           'window': 45, 'poster': 70, 'lift': 47, 'vent': 48, 'fountain': 49, 'door_handle': 50, 'monitor': 51,
           'board': 52,
         # 'floor/walking_path': 53,
          'machines': 54,
           # 'child': 55,
           'trash_bin': 56, 'shelf': 57, 'car': 58,
           'bicycle/scooter': 59, 'bag': 60, 'big_vehicle': 61, 'chair/sofa': 62,
           'pedestrian': 63,
           'trolley': 64,
           'pole/trafficcone': 65,
           'bicyclist/rider': 66,
           'table': 67, 'skateboard/segway/hoverboard': 68,
           'big_socket/button': 69, 'sign': 71, 'decoration': 72,
           # 'background': 73
           }

age_corresponding={ 'Childhood':'child',
'Adolescence' : 'adolescent',
'Young_Adulthood' : 'young',
'Middle_Adulthood' : 'middle-aged',
'Late_Adulthood' : 'elderly',
'impossible' : 'impossible'
    }

VQA_corresponding={ 'age':'age',
'race' : 'race',
'gender' : 'gender',
'action' : 'action',
'distance' : 'distance relative to me(robot)',
'SR_Robot_Ref' : 'spatial position relative to me(robot)',
'hhi' : 'social interaction',
'hhg' : 'geometry between people',
'hoi' : 'interaction between person and object',
'hoG' : 'geometry between person and object',

    }
object_corresponding={'glass': 'glass', 'wall': 'wall', 'door': 'door', 'road': 'road', 'terrain': 'terrain', 'sky': 'sky', 'vegetation': 'vegetation', 'column': 'column',
           'building': 'building', 'stair': 'stair', 'ceiling': 'ceiling', 'barrier/fence': 'fence', 'crutch': 'crutch', 'fire_extinguisher': 'fire extinguisher',
           'airpot': 'airpot', 'controller': 'controller', 'umbrella': 'umbrella', 'animal': 'animal', 'carpet': 'carpet', 'cabinet': 'cabinet', 'jacket': 'jacket',
           'emergency_pole': 'emergency_pole', 'helmet': 'helmet', 'peripheral': 'peripheral', 'curtain': 'curtain', 'box':'box', 'document': 'document',
           'television': 'television', 'light_pole': 'light pole', 'manhole': 'manhole', 'phone': 'phone', 'ladder': 'ladder', 'waterbottle': 'water bottle',
           'picture_frames': 'picture_frames', 'standee': 'standee', 'store_sign': 'store sign', 'statues': 'statues', 'cargo': 'cargo', 'tableware': 'tableware',
           'accessory': 'accessory', 'other': 'other', 'clock': 'clock', 'golfcart': 'golfcart', 'wall_panel': 'wall panel', 'hanging_light': 'hanging light',
           'window': 'window', 'poster': 'poster', 'lift': 'lift', 'vent': 'vent', 'fountain': 'fountain', 'door_handle': 'door handle', 'monitor': 'monitor',
           'board': 'board', 'floor/walking_path': 'floor', 'machines': 'laptop/machines', 'child': 'child', 'trash_bin': 'trash bin', 'shelf': 'shelf', 'car': 'car',
           'bicycle/scooter': 'bicycle', 'bag': 'bag', 'big_vehicle': 'big vehicle', 'chair/sofa': 'chair', 'pedestrian':'pedestrian', 'trolley': 'trolley',
           'pole/trafficcone': 'pole', 'bicyclist/rider': 'bicyclist', 'table': 'table', 'skateboard/segway/hoverboard':'skateboard',
           'big_socket/button': 'big socket', 'sign': 'sign', 'decoration': 'decoration', 'background': 'background',"Not found":''}

# object_specific_corresponding=['glass', 'door', 'terrain', 'vegetation', 'column',
#            'building', 'stair', 'barrier/fence', 'crutch', 'fire_extinguisher',
#            'airpot', 'controller', 'umbrella', 'animal', 'carpet', 'cabinet', 'jacket',
#            'emergency_pole', 'helmet', 'peripheral', 'curtain', 'box', 'document',
#            'television', 'light_pole', 'manhole', 'phone', 'ladder', 'waterbottle',
#            'picture_frames', 'standee', 'store_sign', 'statues', 'cargo', 'tableware',
#            'accessory', 'other', 'clock', 'golfcart', 'wall_panel', 'hanging_light',
#            'window', 'poster', 'lift', 'vent', 'fountain', 'door_handle', 'monitor',
#            'board', 'machines', 'trash_bin', 'shelf', 'car',
#            'bicycle/scooter', 'bag', 'big_vehicle', 'chair/sofa', 'trolley',
#            'pole/trafficcone', 'bicyclist/rider', 'table', 'skateboard/segway/hoverboard',
#            'big_socket/button', 'sign', 'decoration', 'background']

object_specific_corresponding=[
# "pedestrian",
"chair/sofa",
"bag",
"table",
"machines",
"door",
"hanging_light",
# 8.  glass
# 9.  vegetation
"board",
"sign",
# 12. wall
"barrier/fence",
# "floor/walking_path",
# "ceiling",
"bicycle/scooter",
"light_pole",
"trash_bin",
"pole/trafficcone",
"building",
"tableware",
# 22. column
"cabinet",
"shelf",
"window",
# 26. terrain
"stair",
"big_socket/button",
"television",
# 30. sky
"waterbottle",
"picture_frames",
"box",
"car",
"monitor",
# "bicyclist/rider",
"controller",
'manhole',
'jacket',
'decoration',
"standee",
# 42. road
"phone",
"poster",
"store_sign",
"carpet",
"trolley",
# "other",
"document",
# "child",
"airpot",
"cargo",
# "peripheral",
"vent",
"skateboard/segway/hoverboard",
"clock",
"door_handle",
"statues",
"helmet",
"curtain",
"golfcart",
"fountain",
"wall_panel",
"emergency_pole",
"accessory",
"crutch",
"big_vehicle",
"fire_extinguisher",
"umbrella",
"lift",
"animal",
 'ladder']


Options={'age': ['child', 'adolescent', 'young', 'middle-aged', 'elderly'],
            "race": ['White', 'Black', 'Asian', 'others'],
            "gender": ['male', 'female'],
            "action": all_human_pose_action ,
            "distance": ["very close", "close", "moderate", "far", "very far"],
            "hhi": HUMAN_HUMAN_INTERACTION,
            "hhg":GEOMETRY,
            "hoi": ['holding cup','carrying bag', 'walking on the floor'],
            "aim": SOCIAL_AIMS,
            'category': OBJECTS,
            'obj_dis_robot':["very close", "close", "moderate", "far", "very far"],
            'SR_Robot_Ref':GEOMETRY,
}