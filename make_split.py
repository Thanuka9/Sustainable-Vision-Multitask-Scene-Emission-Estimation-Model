# make_split.py

import random

input_list = "all_images.txt"   # your single list file
train_out = "train.txt"
val_out = "val.txt"
val_ratio = 0.2                 # 20% validation

with open(input_list, "r") as f:
    lines = [line.strip() for line in f if line.strip()]

random.shuffle(lines)

n_total = len(lines)
n_val = int(n_total * val_ratio)

val_lines = lines[:n_val]
train_lines = lines[n_val:]

with open(train_out, "w") as f:
    f.write("\n".join(train_lines))

with open(val_out, "w") as f:
    f.write("\n".join(val_lines))

print(f"Total: {n_total}, Train: {len(train_lines)}, Val: {len(val_lines)}")
