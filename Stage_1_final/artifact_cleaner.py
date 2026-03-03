# offline_cleaner_save_mask.py
import os
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

# CONFIG
SOURCE = r"D:\own_split"        # source root (split or unsplit)
DEST   = r"D:\own_cleaned_mask" # destination root
IMG_SIZE = 224                  # final saved size (optional)
BLACK_THRESH = 6                # pixel intensity <= treated as black
GREEN_H_LOW = 35
GREEN_H_HIGH = 85

Path(DEST).mkdir(parents=True, exist_ok=True)

def process_and_save(src_path, dst_img_path, dst_mask_path):
    try:
        pil = Image.open(src_path).convert("RGB")
    except Exception as e:
        print("skip", src_path, e)
        return

    img = np.array(pil)  # RGB

    # --- 1) turn green overlay region into pure black ---
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower = np.array([GREEN_H_LOW, 40, 40], dtype=np.uint8)
    upper = np.array([GREEN_H_HIGH, 255, 255], dtype=np.uint8)
    green_mask = cv2.inRange(hsv, lower, upper)  # 0 or 255

    if green_mask.sum() > 0:
        img[green_mask > 0] = (0, 0, 0)

    # --- 2) compute non-black mask (valid tissue) ---
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    non_black_mask = (gray > BLACK_THRESH).astype(np.uint8)  # 1 for valid

    # If too few valid pixels, try relaxed threshold
    if non_black_mask.sum() < 50:
        non_black_mask = (gray > (BLACK_THRESH//2)).astype(np.uint8)

    # --- 3) optional: crop to non-black bounding box (comment out if not wanted) ---
    ys, xs = np.where(non_black_mask)
    if ys.size and xs.size:
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        # add small padding
        pad = int(min(img.shape[:2]) * 0.02)
        y0 = max(0, y0 - pad); y1 = min(img.shape[0]-1, y1 + pad)
        x0 = max(0, x0 - pad); x1 = min(img.shape[1]-1, x1 + pad)
        img = img[y0:y1+1, x0:x1+1]
        non_black_mask = non_black_mask[y0:y1+1, x0:x1+1]

    # --- 4) resize to IMG_SIZE square and save ---
    img_out = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    mask_out = cv2.resize(non_black_mask.astype(np.uint8), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

    # save image and mask
    cv2.imwrite(dst_img_path, cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR))
    # mask as binary png (0 or 255)
    cv2.imwrite(dst_mask_path, (mask_out * 255).astype(np.uint8))


def process_folder_tree(src_root, dst_root):
    # If dataset already has splits (train/val/test)
    splits = ["train", "val", "test"]
    if all(os.path.exists(os.path.join(src_root, s)) for s in splits):
        print("Detected split dataset:", src_root)
        for s in splits:
            for cls in os.listdir(os.path.join(src_root, s)):
                src_cls = os.path.join(src_root, s, cls)
                if not os.path.isdir(src_cls):
                    continue
                dst_cls = os.path.join(dst_root, s, cls)
                os.makedirs(dst_cls, exist_ok=True)
                for fname in tqdm(sorted(os.listdir(src_cls)), desc=f"{s}/{cls}"):
                    if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue
                    src_path = os.path.join(src_cls, fname)
                    dst_img_path = os.path.join(dst_cls, fname)
                    base, ext = os.path.splitext(fname)
                    dst_mask_path = os.path.join(dst_cls, f"{base}_mask.png")
                    process_and_save(src_path, dst_img_path, dst_mask_path)
    else:
        print("Detected unsplit dataset:", src_root)
        for cls in os.listdir(src_root):
            src_cls = os.path.join(src_root, cls)
            if not os.path.isdir(src_cls):
                continue
            dst_cls = os.path.join(dst_root, cls)
            os.makedirs(dst_cls, exist_ok=True)
            for fname in tqdm(sorted(os.listdir(src_cls)), desc=f"{cls}"):
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                src_path = os.path.join(src_cls, fname)
                dst_img_path = os.path.join(dst_cls, fname)
                base, ext = os.path.splitext(fname)
                dst_mask_path = os.path.join(dst_cls, f"{base}_mask.png")
                process_and_save(src_path, dst_img_path, dst_mask_path)


if __name__ == "__main__":
    process_folder_tree(SOURCE, DEST)
    print("Done. Cleaned images and masks saved to:", DEST)