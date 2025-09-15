# prepare_yolo_segmentation_with_split.py
import os
import json
import random
import argparse
from pathlib import Path
import cv2
import numpy as np

# ---------------- utilities ----------------
def find_mask_for_image(img_name, masks_dir):
    base = Path(img_name).stem
    for ext in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'):
        cand = masks_dir / (base + ext)
        if cand.exists():
            return str(cand)
    return None

def letterbox_resize_pair(img, mask, size):
    """Resize image and mask to (size,size) preserving aspect ratio and padding with black/0.
       img: BGR image (H,W,3)
       mask: single-channel mask (H,W) or None
    """
    h0, w0 = img.shape[:2]
    if w0 == 0 or h0 == 0:
        raise ValueError("Invalid image with zero size")
    scale = size / max(h0, w0)
    new_w, new_h = int(round(w0 * scale)), int(round(h0 * scale))
    # choose interpolation
    img_resized = cv2.resize(img, (new_w, new_h),
                             interpolation=cv2.INTER_LINEAR if img.ndim == 3 else cv2.INTER_NEAREST)
    if mask is not None:
        mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    else:
        mask_resized = None

    # compute padding
    top = (size - new_h) // 2
    bottom = size - new_h - top
    left = (size - new_w) // 2
    right = size - new_w - left

    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))
    if mask_resized is not None:
        mask_padded = cv2.copyMakeBorder(mask_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    else:
        mask_padded = None

    return img_padded, mask_padded, (scale, left, top)

def polylines_from_contour(cnt):
    # convert contour to list of x,y floats
    pts = cnt.reshape(-1, 2)
    return pts.tolist()

# ---------------- main extraction & writing ----------------
def extract_and_write(images_list, images_dir, masks_dir,
                      out_images_dir, out_masks_dir, out_labels_dir,
                      images_meta, annotations_list, start_img_id, start_ann_id,
                      size=640, min_area=50, class_id=0):
    img_id = start_img_id
    ann_id = start_ann_id

    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_masks_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    for im_file in images_list:
        im_path = images_dir / im_file
        mask_path = find_mask_for_image(im_file, masks_dir)
        if mask_path is None:
            print(f"WARNING: mask not found for {im_file} — skipping")
            continue

        img = cv2.imread(str(im_path))
        if img is None:
            print(f"WARNING: failed to read image {im_path} — skipping")
            continue

        # read mask (keep channels if any)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            print(f"WARNING: failed to read mask {mask_path} — skipping")
            continue

        # ensure mask is single-channel
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # resize both with letterbox (keeps alignment)
        img_r, mask_r, (scale, pad_x, pad_y) = letterbox_resize_pair(img, mask, size)

        h, w = img_r.shape[:2]  # should be size,size

        # save resized image and mask
        out_img_file = out_images_dir / im_file
        cv2.imwrite(str(out_img_file), img_r)
        out_mask_file = out_masks_dir / (Path(im_file).stem + ".png")  # masks as png
        # threshold mask -> binary
        _, mask_bin = cv2.threshold(mask_r, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(str(out_mask_file), mask_bin)

        # find contours on the resized mask
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        label_lines = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            seg_pts = polylines_from_contour(cnt)  # list of [x,y]
            # if not enough points -> skip
            if len(seg_pts) < 3:
                continue
            # flatten and normalize to [0,1]
            norm = []
            for (x, y) in seg_pts:
                # x and y are already in resized image coords (including padding)
                xn = float(x) / w
                yn = float(y) / h
                # clamp just in case
                xn = max(0.0, min(1.0, xn))
                yn = max(0.0, min(1.0, yn))
                norm.append(xn)
                norm.append(yn)
            # YOLO polygon line: class followed by normalized coords
            # write with 6 decimal places
            coord_strs = [f"{v:.6f}" for v in norm]
            line = " ".join([str(class_id)] + coord_strs)
            label_lines.append(line)

            # Also add to COCO-style annotations
            seg_flat = [float(coord) for pair in seg_pts for coord in pair]  # polygon in pixels (resized)
            x, y, bw, bh = cv2.boundingRect(cnt)
            ann = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": class_id + 1,   # COCO category ids usually start at 1
                "segmentation": [seg_flat],
                "area": float(area),
                "bbox": [int(x), int(y), int(bw), int(bh)],
                "iscrowd": 0
            }
            annotations_list.append(ann)
            ann_id += 1

        # write label file (can be empty if no instance passed min_area)
        label_path = out_labels_dir / (Path(im_file).stem + ".txt")
        with open(label_path, "w", encoding="utf8") as f:
            for ln in label_lines:
                f.write(ln + "\n")

        # append image meta for COCO (using resized dimensions)
        images_meta.append({
            "id": img_id,
            "file_name": im_file,
            "height": h,
            "width": w
        })
        img_id += 1

    return img_id, ann_id

def write_coco_json(out_json_path, images, annotations, categories):
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, 'w', encoding='utf8') as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)
    print(f"Wrote {out_json_path} ({len(images)} images, {len(annotations)} annotations)")

def visualize_sample(images_dir, masks_dir, sample_images, out_dir, alpha=0.5):
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname in sample_images:
        im_path = images_dir / fname
        mask_path = find_mask_for_image(fname, masks_dir)
        img = cv2.imread(str(im_path))
        if img is None or mask_path is None:
            continue
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        _, bin_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay = img.copy()
        cv2.drawContours(overlay, contours, -1, (0,255,0), thickness=cv2.FILLED)
        out = cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)
        cv2.drawContours(out, contours, -1, (0,200,0), thickness=2)
        cv2.imwrite(str(out_dir / fname), out)
    print(f"Saved {len(sample_images)} visualizations to {out_dir}")

# ---------------- CLI / main ----------------
def main(args):
    images_dir = Path(args.images)
    masks_dir = Path(args.masks)
    out_dir = Path(args.out_dir)
    random.seed(args.seed)

    # gather image files
    files = sorted([p.name for p in images_dir.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp','.tif','.tiff')])
    if not files:
        print("No image files found. Exiting.")
        return

    # shuffle and split
    random.shuffle(files)
    if args.val_size > 0:
        n_val = max(1, int(len(files) * args.val_size))
        val_files = files[:n_val]
        train_files = files[n_val:]
    else:
        train_files = files
        val_files = []

    print(f"Total images: {len(files)} -> train: {len(train_files)}, val: {len(val_files)}")

    # prepare output folder structure
    images_train_out = out_dir / "images" / "train"
    images_val_out = out_dir / "images" / "val"
    masks_train_out = out_dir / "masks" / "train"
    masks_val_out = out_dir / "masks" / "val"
    labels_train_out = out_dir / "labels" / "train"
    labels_val_out = out_dir / "labels" / "val"
    annotations_out = out_dir / "annotations"
    visual_out = out_dir / "visual_check"
    for p in [images_train_out, images_val_out, masks_train_out, masks_val_out, labels_train_out, labels_val_out, annotations_out, visual_out]:
        p.mkdir(parents=True, exist_ok=True)

    # process train
    print("Processing train split...")
    train_images_meta = []
    train_annotations = []
    next_img_id, next_ann_id = extract_and_write(train_files, images_dir, masks_dir,
                                                 images_train_out, masks_train_out, labels_train_out,
                                                 train_images_meta, train_annotations,
                                                 start_img_id=1, start_ann_id=1,
                                                 size=args.size, min_area=args.min_area, class_id=0)

    # process val
    val_images_meta = []
    val_annotations = []
    if val_files:
        print("Processing val split...")
        _, _ = extract_and_write(val_files, images_dir, masks_dir,
                                 images_val_out, masks_val_out, labels_val_out,
                                 val_images_meta, val_annotations,
                                 start_img_id=next_img_id, start_ann_id=next_ann_id,
                                 size=args.size, min_area=args.min_area, class_id=0)

    # write COCO JSONs (optional but useful)
    categories = [{"id": 1, "name": args.class_name}]
    write_coco_json(annotations_out / "instances_train.json", train_images_meta, train_annotations, categories)
    if val_files:
        write_coco_json(annotations_out / "instances_val.json", val_images_meta, val_annotations, categories)

    # write data.yaml for ultralytics (point to folders)
    yaml_path = out_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf8") as f:
        f.write(f"train: {str(images_train_out.resolve())}\n")
        f.write(f"val: {str(images_val_out.resolve()) if val_files else str(images_train_out.resolve())}\n")
        f.write(f"nc: 1\n")
        f.write(f"names: ['{args.class_name}']\n")
    print(f"Wrote data.yaml to {yaml_path}")

    # visual check using original images (not the resized ones) for quick sanity
    sample = (train_files[:min(args.num_visuals, len(train_files))] +
              (val_files[:min(args.num_visuals, len(val_files))] if val_files else []))
    visualize_sample(images_dir, masks_dir, sample, visual_out, alpha=0.5)

    print("All done.")
    print(f"Dataset saved to: {out_dir}")
    print("Structure example:")
    print(f"  {out_dir}/images/train  (images)")
    print(f"  {out_dir}/images/val")
    print(f"  {out_dir}/masks/train   (png masks)")
    print(f"  {out_dir}/masks/val")
    print(f"  {out_dir}/labels/train  (yolo polygon .txt files)")
    print(f"  {out_dir}/labels/val")
    print(f"  {out_dir}/annotations  (COCO jsons)")
    print(f"  {out_dir}/data.yaml")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True, help="path to images folder (e.g. D:\\Project\\segmented-images\\images)")
    p.add_argument("--masks", required=True, help="path to masks folder (e.g. D:\\Project\\segmented-images\\masks)")
    p.add_argument("--out_dir", default="D:\\Project\\segmented-images\\prepared", help="output folder")
    p.add_argument("--class_name", default="object", help="class name for single class")
    p.add_argument("--val_size", type=float, default=0.2, help="fraction to use for validation (0 -> no split)")
    p.add_argument("--min_area", type=int, default=50, help="minimum contour area (in px) to keep as an instance")
    p.add_argument("--num_visuals", type=int, default=8, help="how many sample visual checks to save")
    p.add_argument("--size", type=int, default=640, help="output square size (default 640)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args)
