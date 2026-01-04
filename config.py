# config.py
# ------------------------------------------------------------------------------
# Configuration file for the Sustainable Vision Project.
# This file defines constants and mappings used across the dataset and training scripts.
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Attribute Definitions
# ------------------------------------------------------------------------------
# The model predicts these 4 binary attributes for each image.
# The order here must match the output vector order from the model's attribute head.
ATTRIBUTE_ORDER = ["Indoor", "Crowded", "Bright_Lights", "High_Tech_Equipment"]

# ------------------------------------------------------------------------------
# Model Constants
# ------------------------------------------------------------------------------
NUM_SCENES = 365          # The Places365 dataset contains 365 distinct scene classes.
NUM_ATTRIBUTES = len(ATTRIBUTE_ORDER) # Number of binary attributes to predict (4).

# ------------------------------------------------------------------------------
# Weak Supervision Mappings
# ------------------------------------------------------------------------------
# Since the Places365 dataset doesn't have attribute labels by default, we use
# "weak supervision". This means we manually define heuristics to assign attribute
# labels based on the scene class name.
#
# SCENE_ATTR_MAPPING:
#   A dictionary where the key is the scene name (e.g., "cinema") and the value
#   is a dictionary of attribute values (0 or 1).
#   If a scene is in this list, its attributes are explicitly set as defined.
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

# INDOOR_SCENES:
#   A set of scene names that are considered "Indoor".
#   This is used as a fallback heuristic: if a scene is in this set,
#   the "Indoor" attribute is set to 1.
#   These names must match the class names in the Places365 dataset.
INDOOR_SCENES = {
    "cinema", "gym", "arcade", "server_room",
    "convention_center", "mall", "studio", "stage",
    "theater", "classroom", "office", "airport_terminal",
    "train_station", "living_room", "bedroom",
    "corridor", "library", "dining_room", "conference_room"
}
