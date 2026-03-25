import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw


# 10 distinct bright colors (R,G,B), 0..255
COLOR_LIST: List[Tuple[int, int, int]] = [
    (255, 0, 0),     # red
    (0, 255, 0),     # green
    (0, 0, 255),     # blue
    (255, 255, 0),   # yellow
    (255, 0, 255),   # magenta
    (0, 255, 255),   # cyan
    (255, 128, 0),   # orange
    (128, 0, 255),   # purple
    (0, 128, 255),   # sky-ish
    (128, 255, 0),   # lime-ish
]


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_class_dirs(split_dir: Path) -> List[str]:
    classes = [p.name for p in split_dir.iterdir() if p.is_dir()]
    classes.sort()
    return classes


def build_color_maps(classes: List[str]) -> Tuple[Dict[str, Tuple[int, int, int]], Dict[str, Tuple[int, int, int]]]:
    if len(classes) != 10:
        raise ValueError(f"Expected 10 classes for Imagenette, got {len(classes)}: {classes}")

    # Correct mapping
    class_to_rgb = {cls: COLOR_LIST[i] for i, cls in enumerate(classes)}
    # Swapped mapping (shift by +1 mod 10)
    class_to_rgb_swapped = {cls: COLOR_LIST[(i + 1) % len(classes)] for i, cls in enumerate(classes)}
    return class_to_rgb, class_to_rgb_swapped


def overlay_square(img: Image.Image, rgb: Tuple[int, int, int], square: int, x: int, y: int) -> Image.Image:
    """
    Draws a filled square on the image and returns a new image.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    out = img.copy()
    draw = ImageDraw.Draw(out)
    # Rectangle is inclusive at the end in PIL; this makes the square exactly `square` pixels.
    draw.rectangle([x, y, x + square - 1, y + square - 1], fill=rgb)
    return out


def process_split(
    split: str,
    src_root: Path,
    dst_root: Path,
    class_to_rgb: Dict[str, Tuple[int, int, int]],
    square: int,
    x: int,
    y: int,
    overwrite: bool,
) -> None:
    src_split = src_root / split
    dst_split = dst_root / split
    dst_split.mkdir(parents=True, exist_ok=True)

    classes = list_class_dirs(src_split)
    for cls in classes:
        (dst_split / cls).mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_skipped = 0

    for cls in classes:
        src_cls_dir = src_split / cls
        dst_cls_dir = dst_split / cls

        for p in src_cls_dir.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in IMG_EXTS:
                continue

            rel = p.relative_to(src_cls_dir)
            out_path = dst_cls_dir / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if out_path.exists() and not overwrite:
                n_skipped += 1
                continue

            try:
                img = Image.open(p)
                img2 = overlay_square(img, class_to_rgb[cls], square=square, x=x, y=y)
                # Keep JPEGs JPEG; otherwise save as PNG.
                if out_path.suffix.lower() in {".jpg", ".jpeg"}:
                    img2.save(out_path, quality=95, subsampling=0)
                else:
                    # if original is jpeg but path is something else, preserve suffix
                    img2.save(out_path)
                n_written += 1
            except Exception as e:
                print(f"[WARN] Failed on {p}: {e}")

    print(f"[{split}] written: {n_written}, skipped: {n_skipped}")


def write_mapping_json(dst_root: Path, mapping: Dict[str, Tuple[int, int, int]], filename: str) -> None:
    out = {cls: list(rgb) for cls, rgb in mapping.items()}
    with open(dst_root / filename, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote mapping file: {dst_root / filename}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, default="data/imagenette2-320", help="Source dataset root (contains train/val)")
    ap.add_argument("--dst", type=str, default="data/imagenette2-320-colorcue", help="Destination dataset root")
    ap.add_argument("--square", type=int, default=16, help="Square size in pixels")
    ap.add_argument("--x", type=int, default=0, help="Square top-left x")
    ap.add_argument("--y", type=int, default=0, help="Square top-left y")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    ap.add_argument("--make-swapped", action="store_true", help="Also create swapped-color dataset variant")
    args = ap.parse_args()

    src_root = Path(args.src).resolve()
    dst_root = Path(args.dst).resolve()

    if not (src_root / "train").exists() or not (src_root / "val").exists():
        raise FileNotFoundError(f"Source root must contain train/ and val/: {src_root}")

    # Determine class order from train split (Imagenette uses same classes for val)
    classes = list_class_dirs(src_root / "train")
    class_to_rgb, class_to_rgb_swapped = build_color_maps(classes)

    # Create correct-color dataset
    dst_root.mkdir(parents=True, exist_ok=True)
    write_mapping_json(dst_root, class_to_rgb, "class_to_rgb.json")

    print(f"\nCreating color-cue dataset:")
    print(f"  src: {src_root}")
    print(f"  dst: {dst_root}")
    print(f"  square: {args.square}px at ({args.x},{args.y})")
    print(f"  overwrite: {args.overwrite}\n")

    process_split("train", src_root, dst_root, class_to_rgb, args.square, args.x, args.y, args.overwrite)
    process_split("val", src_root, dst_root, class_to_rgb, args.square, args.x, args.y, args.overwrite)

    # Optionally create swapped-color variant (for adversarial evaluation)
    if args.make_swapped:
        swapped_root = Path(str(dst_root) + "-swapped").resolve()
        swapped_root.mkdir(parents=True, exist_ok=True)
        write_mapping_json(swapped_root, class_to_rgb_swapped, "class_to_rgb_swapped.json")

        print(f"\nCreating swapped-color dataset variant:")
        print(f"  dst: {swapped_root}\n")

        process_split("train", src_root, swapped_root, class_to_rgb_swapped, args.square, args.x, args.y, args.overwrite)
        process_split("val", src_root, swapped_root, class_to_rgb_swapped, args.square, args.x, args.y, args.overwrite)

    print("\nDone.")


if __name__ == "__main__":
    main()
