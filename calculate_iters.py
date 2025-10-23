import numpy as np
import math
import os

# ⚠️ Change this path if needed
EXTRA = r"D:\gi-project\gi_dino_dataset\extra"

entries_path = os.path.join(EXTRA, "entries-TRAIN.npy")

if not os.path.exists(entries_path):
    print(f"❌ File not found: {entries_path}")
    exit()

arr = np.load(entries_path, allow_pickle=True)
n_images = len(arr)

batch_size_per_gpu = 16   # ← adjust to what you plan to use
world_size = 1            # ← set to >1 if using multi-GPU

global_batch = batch_size_per_gpu * world_size
iters_per_epoch = math.ceil(n_images / global_batch)

print("\n✅ TRAIN IMAGES:", n_images)
print("✅ BATCH SIZE (Global):", global_batch)
print("✅ ITERATIONS PER EPOCH:", iters_per_epoch)



'''
FINDINGS:
    ✅ TRAIN IMAGES: 1068534
    ✅ BATCH SIZE (Global): 16
    ✅ ITERATIONS PER EPOCH: 66784
'''