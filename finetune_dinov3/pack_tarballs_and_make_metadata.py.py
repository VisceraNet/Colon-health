#!/usr/bin/env python3
"""
pack_tarballs_and_make_metadata.py

Create uncompressed tarballs and DINOv3-style extra/ metadata (entries/class-ids/class-names)
from a directory of extracted video-frame folders.

Usage example:
  python pack_tarballs_and_make_metadata.py \
    --input-root /home/dev/projects/gi-project/data/unlabeled_data \
    --output-root /home/dev/projects/gi-project/gi_dino_dataset \
    --train 60 --val 8 --test 6 \
    --max-side 1024 \
    --force

Outputs:
  <output_root>/tarballs/<class_id>.tar        (one .tar per video folder used in any split)
  <output_root>/blocks/<class_id>.log         (block offset log for each tarball)
  <output_root>/extra/entries-TRAIN.npy       (structured entries array)
  <output_root>/extra/class-ids-TRAIN.npy
  <output_root>/extra/class-names-TRAIN.npy
  (same for VAL and TEST)
Notes:
  - Tar files are written in plain 'ustar' format (no compression) so offsets are stable.
  - Images are re-encoded as JPEG (RGB) and resized to keep long side <= max_side.
"""

import argparse
import os
import tarfile
from io import BytesIO
from PIL import Image
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
import math

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")


def list_video_dirs(path: str) -> List[str]:
    items = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
    return items


def is_image_file(name: str) -> bool:
    return name.lower().endswith(IMAGE_EXTS)


def resize_and_encode_jpeg(img_path: str, max_side: int = 1024, quality: int = 90) -> bytes:
    """
    Open image, convert to RGB, resize keeping aspect ratio so that the long side <= max_side,
    and return JPEG bytes.
    """
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        # Resize preserving aspect ratio
        w, h = im.size
        if max(w, h) > max_side:
            if w >= h:
                new_w = max_side
                new_h = int(round(h * (max_side / float(w))))
            else:
                new_h = max_side
                new_w = int(round(w * (max_side / float(h))))
            im = im.resize((new_w, new_h), Image.LANCZOS)
        bio = BytesIO()
        im.save(bio, format="JPEG", quality=quality, optimize=True)
        data = bio.getvalue()
        return data


def pack_single_tar(video_dir: str, tar_path: str, max_side: int = 1024, q: int = 90) -> List[Tuple[str,int,int]]:
    """
    Pack all image files under video_dir into tar_path (uncompressed).
    Returns list of tuples: (filename_inside_tar, start_offset, end_offset)
    where offsets are byte offsets in the tar file (start inclusive, end exclusive).
    """
    ensure_parent = os.path.dirname(tar_path)
    os.makedirs(ensure_parent, exist_ok=True)

    with tarfile.open(tar_path, "w") as tar:
        records = []
        image_files = sorted([f for f in os.listdir(video_dir) if is_image_file(f)])
        for img_name in image_files:
            img_path = os.path.join(video_dir, img_name)
            # resize+encode
            data = resize_and_encode_jpeg(img_path, max_side=max_side, quality=q)
            buf = BytesIO(data)
            buf.seek(0)
            # create tarinfo
            tarinfo = tarfile.TarInfo(name=img_name)
            tarinfo.size = len(data)
            # optional metadata: mtime
            try:
                tarinfo.mtime = int(os.path.getmtime(img_path))
            except Exception:
                tarinfo.mtime = 0
            # compute start offset (current fileobj position)
            offset_before = tar.fileobj.tell()
            block_offset = offset_before // 512
            start_offset = block_offset * 512
            tar.addfile(tarinfo, fileobj=buf)
            offset_after = tar.fileobj.tell()
            end_offset = offset_after  # exclusive
            records.append((img_name, int(start_offset), int(end_offset)))
    return records


def build_tarballs_and_metadata(input_root: str, output_root: str, train_count:int, val_count:int, test_count:int, max_side:int=1024, force:bool=False):
    # gather video dirs
    all_videos = list_video_dirs(input_root)
    if len(all_videos) < (train_count + val_count + test_count):
        raise SystemExit(f"Need at least {train_count+val_count+test_count} video dirs; found {len(all_videos)}")
    train_dirs = all_videos[:train_count]
    val_dirs = all_videos[train_count: train_count+val_count]
    test_dirs = all_videos[train_count+val_count: train_count+val_count+test_count]
    splits = {"TRAIN": train_dirs, "VAL": val_dirs, "TEST": test_dirs}

    tarballs_root = os.path.join(output_root, "tarballs")
    blocks_root = os.path.join(output_root, "blocks")
    extra_root = os.path.join(output_root, "extra")
    os.makedirs(tarballs_root, exist_ok=True)
    os.makedirs(blocks_root, exist_ok=True)
    os.makedirs(extra_root, exist_ok=True)

    # We'll create tarballs for every class_id used in any split in the same tarballs_root.
    class_order = []  # map class_index -> class_id
    entries_by_split = {"TRAIN": [], "VAL": [], "TEST": []}
    class_ids_by_split = {"TRAIN": [], "VAL": [], "TEST": []}
    class_names_by_split = {"TRAIN": [], "VAL": [], "TEST": []}

    # Keep a deterministic class_index mapping: all unique class_ids in the order we process them
    unique_classes = []

    # Process splits and create tarballs/logs
    for split_label, video_list in splits.items():
        for vid in tqdm(video_list, desc=f"Tarball packing {split_label}"):
            src_dir = os.path.join(input_root, vid)
            class_id = vid
            tar_path = os.path.join(tarballs_root, f"{class_id}.tar")
            blocks_log_path = os.path.join(blocks_root, f"{class_id}.log")

            if os.path.exists(tar_path) and not force:
                # we need to read offsets from existing tar file? easier to rebuild if force set.
                raise SystemExit(f"Tarball {tar_path} already exists. Rerun with --force to overwrite.")
            # pack
            records = pack_single_tar(src_dir, tar_path, max_side=max_side)
            # write blocks log: each line = "<block_index>: <filename>"
            with open(blocks_log_path, "w") as f:
                for (filename, start_offset, end_offset) in records:
                    block_index = start_offset // 512
                    f.write(f"block{block_index:07d}: {filename}\n")
                # final sentinel block index pointing after last file (to calculate end offsets)
                final_block = math.ceil(end_offset / 512)
                f.write(f"block{final_block:07d}: ** Block of NULs **\n")

            # register class if new
            if class_id not in unique_classes:
                unique_classes.append(class_id)
            class_index = unique_classes.index(class_id)

            # build entries (entries are stored relative to tarball "class", format used by ImageNet-like loader)
            for (filename, start_offset, end_offset) in records:
                entries_by_split[split_label].append( (class_index, class_id, start_offset, end_offset, filename) )

            class_ids_by_split[split_label].append(str(class_index))
            class_names_by_split[split_label].append(class_id)

    # Save metadata numpy arrays under extra/
    def save_split_arrays(split_label):
        entries = entries_by_split[split_label]
        # create structured numpy array matching the ImageNet dtype used by DINOv3
        if len(entries) == 0:
            arr = np.empty(0, dtype=[("class_index","<u4"), ("class_id","U1"), ("start_offset","<u4"), ("end_offset","<u4"), ("filename","U1")])
        else:
            max_class_id_len = max(len(e[1]) for e in entries)
            max_filename_len = max(len(e[4]) for e in entries)
            dtype = np.dtype([
                ("class_index", "<u4"),
                ("class_id", f"U{max_class_id_len}"),
                ("start_offset", "<u4"),
                ("end_offset", "<u4"),
                ("filename", f"U{max_filename_len}"),
            ])
            arr = np.empty(len(entries), dtype=dtype)
            for i, (class_index, class_id, start, end, filename) in enumerate(entries):
                arr[i] = (np.uint32(class_index), class_id, np.uint32(start), np.uint32(end), filename)

        entries_path = os.path.join(extra_root, f"entries-{split_label}.npy")
        np.save(entries_path, arr)

        # class-ids and class-names arrays: note — class indices should represent all classes used in that split
        class_ids_arr = np.array(class_ids_by_split[split_label], dtype=object)
        class_names_arr = np.array(class_names_by_split[split_label], dtype=object)
        np.save(os.path.join(extra_root, f"class-ids-{split_label}.npy"), class_ids_arr)
        np.save(os.path.join(extra_root, f"class-names-{split_label}.npy"), class_names_arr)

    save_split_arrays("TRAIN")
    save_split_arrays("VAL")
    save_split_arrays("TEST")

    # Also save a global class-ids/class-names if you want
    np.save(os.path.join(extra_root, "global-class-ids.npy"), np.array(unique_classes, dtype=object))
    np.save(os.path.join(extra_root, "global-class-names.npy"), np.array(unique_classes, dtype=object))

    print("Done. Tarballs root:", tarballs_root)
    print("Blocks root:", blocks_root)
    print("Extra metadata root:", extra_root)
    # summary counts
    print("Counts (images): TRAIN", len(entries_by_split["TRAIN"]), "VAL", len(entries_by_split["VAL"]), "TEST", len(entries_by_split["TEST"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", required=True, help="Path containing extracted video-frame folders (each folder = one video)")
    p.add_argument("--output-root", required=True, help="Output root to create tarballs/, blocks/, extra/")
    p.add_argument("--train", type=int, default=60)
    p.add_argument("--val", type=int, default=8)
    p.add_argument("--test", type=int, default=6)
    p.add_argument("--max-side", type=int, default=1024, help="Max long side when resizing (keeps aspect ratio).")
    p.add_argument("--force", action="store_true", help="Overwrite existing tarballs/logs.")
    args = p.parse_args()
    build_tarballs_and_metadata(args.input_root, args.output_root, args.train, args.val, args.test, max_side=args.max_side, force=args.force)
