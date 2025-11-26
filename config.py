# config.py

# Attribute order for the multi-label output head
ATTRIBUTE_ORDER = ["Indoor", "Crowded", "Bright_Lights", "High_Tech_Equipment"]

NUM_SCENES = 365          # Places365 has 365 scene classes
NUM_ATTRIBUTES = len(ATTRIBUTE_ORDER)

# Weak-supervision mapping: manually chosen heuristics

SCENE_ATTR_MAPPING = {
    # High_Tech_Equipment / bright
    "cinema":              {"High_Tech_Equipment": 1, "Crowded": 0, "Indoor": 1, "Bright_Lights": 1},
    "gym":                 {"High_Tech_Equipment": 1, "Crowded": 0, "Indoor": 1, "Bright_Lights": 0},
    "arcade":              {"High_Tech_Equipment": 1, "Crowded": 1, "Indoor": 1, "Bright_Lights": 1},
    "server_room":         {"High_Tech_Equipment": 1, "Crowded": 0, "Indoor": 1, "Bright_Lights": 0},

    # Crowded locations
    "stadium":             {"High_Tech_Equipment": 0, "Crowded": 1, "Indoor": 0, "Bright_Lights": 1},
    "convention_center":   {"High_Tech_Equipment": 0, "Crowded": 1, "Indoor": 1, "Bright_Lights": 1},
    "market":              {"High_Tech_Equipment": 0, "Crowded": 1, "Indoor": 0, "Bright_Lights": 0},
    "train_station":       {"High_Tech_Equipment": 0, "Crowded": 1, "Indoor": 1, "Bright_Lights": 0},

    # Bright lights / indoor public
    "theater":             {"High_Tech_Equipment": 1, "Crowded": 1, "Indoor": 1, "Bright_Lights": 1},
    "mall":                {"High_Tech_Equipment": 1, "Crowded": 1, "Indoor": 1, "Bright_Lights": 1},
    "studio":              {"High_Tech_Equipment": 1, "Crowded": 0, "Indoor": 1, "Bright_Lights": 1},
    "stage":               {"High_Tech_Equipment": 1, "Crowded": 1, "Indoor": 1, "Bright_Lights": 1},
}

# Rough indoor scene list. Names should match Places365 class names.
INDOOR_SCENES = {
    "cinema", "gym", "arcade", "server_room",
    "convention_center", "mall", "studio", "stage",
    "theater", "classroom", "office", "airport_terminal",
    "train_station", "living_room", "bedroom",
    "corridor", "library", "dining_room", "conference_room"
}
