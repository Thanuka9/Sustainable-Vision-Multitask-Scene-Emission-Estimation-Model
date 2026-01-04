# make_split.py
# ------------------------------------------------------------------------------
# Utility script to split a single list of image files into Train and Validation sets.
# This is useful when you have a raw list of images and need to prepare them for
# dataset loaders that expect separate split files.
# ------------------------------------------------------------------------------

import random

input_list = "all_images.txt"   # Path to your single list file containing all image paths
train_out = "train.txt"         # Output file for training paths
val_out = "val.txt"             # Output file for validation paths
val_ratio = 0.2                 # Ratio of images to use for validation (20%)

# Read all lines from the input file
with open(input_list, "r") as f:
    lines = [line.strip() for line in f if line.strip()]

# Shuffle the list randomly to ensure a good distribution
random.shuffle(lines)

# Calculate split index
n_total = len(lines)
n_val = int(n_total * val_ratio)

# Slice the list into validation and training sets
val_lines = lines[:n_val]
train_lines = lines[n_val:]

# Write the training set to file
with open(train_out, "w") as f:
    f.write("\n".join(train_lines))

# Write the validation set to file
with open(val_out, "w") as f:
    f.write("\n".join(val_lines))

print(f"Total: {n_total}, Train: {len(train_lines)}, Val: {len(val_lines)}")
