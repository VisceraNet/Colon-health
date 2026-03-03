import os
import random
import shutil
from pathlib import Path

# =====================================
# CONFIG
# =====================================
SOURCE_DIR = r"D:\own"
DEST_DIR   = r"D:\own_split"

TRAIN_RATIO = 0.7
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

SEED = 42
random.seed(SEED)

# =====================================
# CREATE FOLDERS
# =====================================
def create_split_folders():
    for split in ["train", "val", "test"]:
        split_path = os.path.join(DEST_DIR, split)
        os.makedirs(split_path, exist_ok=True)

# =====================================
# SPLIT DATASET
# =====================================
def split_dataset():

    classes = os.listdir(SOURCE_DIR)
    classes = [c for c in classes if os.path.isdir(os.path.join(SOURCE_DIR, c))]

    print("Classes found:", classes)

    for cls in classes:

        src_class_path = os.path.join(SOURCE_DIR, cls)
        images = os.listdir(src_class_path)
        images = [img for img in images if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

        random.shuffle(images)

        total = len(images)
        train_end = int(TRAIN_RATIO * total)
        val_end   = train_end + int(VAL_RATIO * total)

        splits = {
            "train": images[:train_end],
            "val":   images[train_end:val_end],
            "test":  images[val_end:]
        }

        for split_name, split_files in splits.items():

            dest_class_path = os.path.join(DEST_DIR, split_name, cls)
            os.makedirs(dest_class_path, exist_ok=True)

            for file in split_files:
                src_file = os.path.join(src_class_path, file)
                dest_file = os.path.join(dest_class_path, file)

                shutil.copy2(src_file, dest_file)

        print(f"{cls}: {len(images)} images split")

    print("\nDataset successfully split and saved to:", DEST_DIR)


# =====================================
# RUN
# =====================================
if __name__ == "__main__":
    create_split_folders()
    split_dataset()