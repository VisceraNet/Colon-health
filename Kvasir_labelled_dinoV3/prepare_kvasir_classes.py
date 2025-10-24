#!/usr/bin/env python3
"""
prepare_kvasir_classes.py

Usage:
    python3 prepare_kvasir_classes.py --zip /path/to/kvasir-capsule-labeled-images.zip --out /home/phil/kvasir_prepared --resize 384

What this does:
- Extracts the zip (and internal .tar/.tar.gz/.tgz/.zip/.gz archives)
- Places images into output_dir/raw/<class_name>/
- For each class, augments images (flips/rotate/color-jitter/brightness/contrast/blur)
  until every class has the same number of images (the maximum class count)
- Splits into train/val/test (80/10/10) under output_dir/split/{train,val,test}/{class}/

Notes:
- Uses Pillow and tqdm. Install with: pip install pillow tqdm
- Designed for large datasets: files are streamed, nothing huge kept in memory.
"""

import argparse
import os
import zipfile
import tarfile
import gzip
import shutil
from pathlib import Path
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import io
import random
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

def is_image_file(p: Path):
    return p.suffix.lower() in IMAGE_EXTS

def safe_makedirs(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def extract_archive_member_bytes(path_bytes: bytes, dest_dir: Path, member_name_hint="file"):
    """
    Try to interpret bytes as an image (PIL) and save with a safe name.
    Returns path of saved image or None.
    """
    try:
        img = Image.open(io.BytesIO(path_bytes))
        img.verify()  # verify integrity
        img = Image.open(io.BytesIO(path_bytes)).convert("RGB")
    except Exception:
        return None
    # create filename
    filename = f"{member_name_hint}.jpg"
    out_path = dest_dir / filename
    i = 0
    while out_path.exists():
        i += 1
        out_path = dest_dir / f"{member_name_hint}_{i}.jpg"
    img.save(out_path, quality=95)
    return out_path

def extract_zip(zip_path: Path, temp_dir: Path):
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(temp_dir)

def try_extract_generic(archive_path: Path, out_base: Path):
    """
    Accepts .zip, .tar, .tar.gz, .tgz, .gz.
    For .gz that wraps a single file: will try to decompress and save if image.
    Returns list of extracted files (Paths).
    """
    extracted = []
    p = archive_path
    if zipfile.is_zipfile(p):
        try:
            with zipfile.ZipFile(p, 'r') as z:
                for member in z.namelist():
                    if member.endswith('/'): continue
                    member_path = Path(member)
                    suffix = member_path.suffix.lower()
                    # If inner is archive, extract to temp
                    if suffix in ('.zip', '.tar', '.gz', '.tgz', '.tar.gz'):
                        # extract member to bytes then handle
                        data = z.read(member)
                        tmp = out_base / (archive_path.stem + "_" + member_path.stem)
                        safe_makedirs(tmp)
                        # write bytes to file and recursively call
                        inner_path = tmp / member_path.name
                        inner_path.write_bytes(data)
                        extracted += try_extract_generic(inner_path, tmp)
                    elif suffix in IMAGE_EXTS:
                        # write image file
                        dest = out_base / member_path.name
                        safe_makedirs(out_base)
                        with open(dest, "wb") as f:
                            f.write(z.read(member))
                        extracted.append(dest)
                    else:
                        # try to parse as image
                        data = z.read(member)
                        tmp = out_base / (member_path.stem + ".jpg")
                        imgp = extract_archive_member_bytes(data, out_base, member_path.stem)
                        if imgp:
                            extracted.append(imgp)
        except Exception as e:
            print("zip extraction failed for", p, e)
    elif tarfile.is_tarfile(p):
        try:
            with tarfile.open(p, 'r:*') as t:
                for member in t.getmembers():
                    if member.isdir(): continue
                    member_path = Path(member.name)
                    suffix = member_path.suffix.lower()
                    fobj = t.extractfile(member)
                    if fobj is None:
                        continue
                    data = fobj.read()
                    if suffix in ('.zip', '.tar', '.tgz', '.tar.gz', '.gz'):
                        tmp = out_base / (archive_path.stem + "_" + member_path.stem)
                        safe_makedirs(tmp)
                        inner_path = tmp / member_path.name
                        inner_path.write_bytes(data)
                        extracted += try_extract_generic(inner_path, tmp)
                    elif suffix in IMAGE_EXTS:
                        safe_makedirs(out_base)
                        dest = out_base / member_path.name
                        with open(dest, "wb") as f:
                            f.write(data)
                        extracted.append(dest)
                    else:
                        imgp = extract_archive_member_bytes(data, out_base, member_path.stem)
                        if imgp:
                            extracted.append(imgp)
        except Exception as e:
            print("tar extraction failed for", p, e)
    elif p.suffix.lower() == ".gz":
        # could be a single gz compressed file (maybe an image or tar inside)
        try:
            with gzip.open(p, 'rb') as gz:
                data = gz.read()
                # try tar detection
                # write to temp and try tar
                try:
                    tmp_path = out_base / (p.stem + "_inner")
                    safe_makedirs(tmp_path)
                    inner_file = tmp_path / (p.stem)
                    inner_file.write_bytes(data)
                    if tarfile.is_tarfile(inner_file):
                        extracted += try_extract_generic(inner_file, tmp_path)
                    elif zipfile.is_zipfile(inner_file):
                        extracted += try_extract_generic(inner_file, tmp_path)
                    else:
                        imgp = extract_archive_member_bytes(data, out_base, p.stem)
                        if imgp:
                            extracted.append(imgp)
                except Exception:
                    imgp = extract_archive_member_bytes(data, out_base, p.stem)
                    if imgp:
                        extracted.append(imgp)
        except Exception as e:
            print("gz extraction failed for", p, e)
    else:
        # unknown file type; attempt to open as image
        try:
            with open(p, "rb") as f:
                data = f.read()
                imgp = extract_archive_member_bytes(data, out_base, p.stem)
                if imgp:
                    extracted.append(imgp)
        except Exception:
            pass
    return extracted

# ---------- Augmentations (PIL-based simple augmentations) ----------
def augment_image(img: Image.Image, mode: int):
    """
    mode selects augmentation:
      0 - horizontal flip
      1 - rotate 90
      2 - rotate 180
      3 - rotate 270
      4 - color jitter (brightness)
      5 - color jitter (contrast)
      6 - color jitter (color)
      7 - gaussian blur
      8 - transpose (flip+rotate)
      9 - small random crop & resize
    """
    if mode == 0:
        return ImageOps.mirror(img)
    if mode == 1:
        return img.rotate(90, expand=True)
    if mode == 2:
        return img.rotate(180, expand=True)
    if mode == 3:
        return img.rotate(270, expand=True)
    if mode == 4:
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(0.7 + random.random() * 0.6)  # [0.7,1.3]
    if mode == 5:
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(0.7 + random.random() * 0.6)
    if mode == 6:
        enhancer = ImageEnhance.Color(img)
        return enhancer.enhance(0.6 + random.random() * 0.9)
    if mode == 7:
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    if mode == 8:
        return ImageOps.mirror(img.rotate(90, expand=True))
    if mode == 9:
        # small crop then resize back
        w, h = img.size
        cw = int(w * (0.8 + random.random() * 0.15))
        ch = int(h * (0.8 + random.random() * 0.15))
        left = random.randint(0, max(0, w - cw))
        top = random.randint(0, max(0, h - ch))
        crop = img.crop((left, top, left + cw, top + ch))
        return crop.resize((w, h), Image.LANCZOS)
    # default no-op
    return img

# ---------- Main pipeline ----------
def main(zip_path: Path, output_dir: Path, resize: int = None, seed: int = 42):
    random.seed(seed)
    safe_makedirs(output_dir)
    tmp_extract = output_dir / "tmp_extracted"
    if tmp_extract.exists():
        shutil.rmtree(tmp_extract)
    safe_makedirs(tmp_extract)

    # 1) Extract main zip to tmp
    print("Extracting main zip:", zip_path)
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            members = z.namelist()
            z.extractall(tmp_extract)
    except Exception as e:
        print("Failed to extract main zip using ZipFile, trying shutil.unpack_archive:", e)
        shutil.unpack_archive(str(zip_path), str(tmp_extract))

    # 2) Find labelled_images folder (or work with everything)
    # Heuristic: folder named 'labelled_images' or 'labelled' or 'labelled_images' inside extracted
    possible_roots = [p for p in tmp_extract.rglob("*") if p.is_dir() and ("label" in p.name.lower())]
    if possible_roots:
        root = possible_roots[0]
        print("Found labelled folder:", root)
    else:
        root = tmp_extract
        print("Using extracted root:", root)

    # 3) For each class folder or for each archive inside root: detect classes
    # Sometimes classes are represented as gz files, named by class
    classes_out = output_dir / "raw"
    safe_makedirs(classes_out)

    # If there are class-named folders directly, copy images
    class_dirs = [p for p in root.iterdir() if p.is_dir()]
    if class_dirs:
        # iterate each folder as class if it contains images or archives
        for c in class_dirs:
            class_name = c.name
            dest = classes_out / class_name
            safe_makedirs(dest)
            # if folder contains archives (.gz etc), extract them
            entries = list(c.rglob("*"))
            for ent in entries:
                if ent.is_file():
                    if is_image_file(ent):
                        # copy image
                        outf = dest / ent.name
                        # avoid duplicate by renaming if exists
                        i = 0
                        base = outf.stem
                        while outf.exists():
                            i += 1
                            outf = dest / f"{base}_{i}{ent.suffix}"
                        shutil.copy2(ent, outf)
                    elif ent.suffix.lower() in {".zip", ".tar", ".gz", ".tgz", ".tar.gz"}:
                        # extract archive into class folder
                        extracted = try_extract_generic(ent, dest)
                        # extracted returns list of file paths saved
            # If no images were found in this dir, skip (likely meta)
            if not any(dest.rglob("*")):
                # remove empty
                try:
                    shutil.rmtree(dest)
                except Exception:
                    pass
    else:
        # No class dirs — maybe there are archives named per class in root
        # Find archives in root: each archive name may encode class name
        for ent in root.iterdir():
            if ent.is_file():
                stem = ent.stem
                # if it's an archive or .gz
                if ent.suffix.lower() in {".zip", ".tar", ".gz", ".tgz", ".tar.gz"}:
                    class_name = stem
                    dest = classes_out / class_name
                    safe_makedirs(dest)
                    extracted = try_extract_generic(ent, dest)
                elif is_image_file(ent):
                    # put into 'unknown' or single class
                    dest = classes_out / "unknown"
                    safe_makedirs(dest)
                    shutil.copy2(ent, dest / ent.name)
    # At this point, classes_out should contain class folders with images
    class_folders = sorted([p for p in classes_out.iterdir() if p.is_dir()])
    print("Detected class folders:", [p.name for p in class_folders])

    # 4) Normalize filenames and optionally resize
    print("Normalizing filenames and resizing (if requested)...")
    for c in tqdm(class_folders):
        files = [p for p in c.iterdir() if p.is_file() and is_image_file(p)]
        for idx, f in enumerate(files):
            try:
                with Image.open(f) as im:
                    im = im.convert("RGB")
                    if resize:
                        im = im.resize((resize, resize), Image.LANCZOS)
                    outname = c / f"img_{idx:06d}.jpg"
                    # avoid overwriting
                    i = 0
                    while outname.exists():
                        i += 1
                        outname = c / f"img_{idx:06d}_{i}.jpg"
                    im.save(outname, quality=95)
                if f.name != outname.name:
                    try:
                        f.unlink()
                    except Exception:
                        pass
            except Exception as e:
                print("Skipping bad image", f, e)

    # refresh class folders and counts
    class_folders = sorted([p for p in classes_out.iterdir() if p.is_dir()])
    class_counts = {p.name: len([q for q in p.iterdir() if q.is_file() and is_image_file(q)]) for p in class_folders}
    print("Initial class counts:", class_counts)

    # 5) Data augmentation to balance classes (simple deterministic augmentations)
    max_count = max(class_counts.values()) if class_counts else 0
    print("Target images per class (max):", max_count)
    augment_modes = list(range(10))  # 0..9 modes

    print("Augmenting minority classes to match max count...")
    for c in tqdm(class_folders):
        imgs = sorted([p for p in c.iterdir() if p.is_file() and is_image_file(p)])
        cur = len(imgs)
        if cur >= max_count:
            continue
        # cycle through existing images and augment
        i = 0
        while cur < max_count:
            src = imgs[i % len(imgs)]
            try:
                with Image.open(src) as im:
                    im = im.convert("RGB")
                    mode = augment_modes[(i // len(imgs)) % len(augment_modes)]
                    aug = augment_image(im, mode)
                    # optionally resize to same target if requested
                    if resize:
                        aug = aug.resize((resize, resize), Image.LANCZOS)
                    outp = c / f"aug_{i:06d}.jpg"
                    j = 0
                    while outp.exists():
                        j += 1
                        outp = c / f"aug_{i:06d}_{j}.jpg"
                    aug.save(outp, quality=95)
                    cur += 1
                    i += 1
            except Exception as e:
                print("Aug error for", src, e)
                i += 1
                if i > len(imgs) * 20:
                    # safety bail
                    break

    # 6) Create split dirs and split 80/10/10 (shuffle deterministic)
    print("Creating train/val/test splits (80/10/10)...")
    split_root = output_dir / "split"
    for s in ("train","val","test"):
        safe_makedirs(split_root / s)

    for c in class_folders:
        files = sorted([p for p in c.iterdir() if p.is_file() and is_image_file(p)])
        random.shuffle(files)
        n = len(files)
        ntrain = int(n * 0.8)
        nval = int(n * 0.1)
        ntest = n - ntrain - nval
        train_files = files[:ntrain]
        val_files = files[ntrain:ntrain+nval]
        test_files = files[ntrain+nval:]
        for subset, flist in [("train", train_files), ("val", val_files), ("test", test_files)]:
            dest_dir = split_root / subset / c.name
            safe_makedirs(dest_dir)
            for f in flist:
                dst = dest_dir / f.name
                # copy to preserve raw
                if not dst.exists():
                    shutil.copy2(f, dst)

    # 7) Summary
    final_counts = {}
    for s in ("train","val","test"):
        sd = split_root / s
        cnt = sum(1 for _ in sd.rglob("*") if _.is_file())
        final_counts[s] = cnt
    print("Done. Final split counts:", final_counts)
    print("Per-class counts (train):")
    for c in class_folders:
        ctrain = len(list((split_root/"train"/c.name).glob("*")))
        cval = len(list((split_root/"val"/c.name).glob("*")))
        ctest = len(list((split_root/"test"/c.name).glob("*")))
        print(f"{c.name}: train={ctrain}, val={cval}, test={ctest}")

    # cleanup temp (optional)
    try:
        shutil.rmtree(tmp_extract)
    except Exception:
        pass
    print("All done. Prepared data at:", split_root)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="Path to the labelled zip (kvasir-capsule-labeled-images.zip)")
    ap.add_argument("--out", required=True, help="Output directory to write raw/ and split/ folders")
    ap.add_argument("--resize", type=int, default=384, help="Resize image short/long side to this square size (optional)")
    ap.add_argument("--seed", type=int, default=42)
    args2 = ap.parse_args()
    main(Path(args2.zip), Path(args2.out), resize=args2.resize, seed=args2.seed)
