"""Prepare paired inputs from the MSRS, TNO, RoadScene and LLVIP archives."""

import argparse
import csv
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LAYOUTS = {
    "msrs-train": ("train/ir", "train/vi", 1083),
    "msrs-test": ("test/ir", "test/vi", 361),
    "roadscene": ("cropinfrared", "crop_LR_visible", 221),
    "llvip": ("infrared/test", "visible/test", 3463),
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def index(archive, suffix):
    images = {}
    parts = tuple(suffix.casefold().split("/"))
    for item in archive.infolist():
        path = PurePosixPath(item.filename.replace(chr(92), "/"))
        if item.is_dir() or path.suffix.casefold() not in IMAGE_SUFFIXES:
            continue
        if tuple(s.casefold() for s in path.parent.parts[-len(parts):]) != parts:
            continue
        key = path.stem.casefold()
        if key in images:
            raise ValueError(f"Duplicate image stem in {suffix}: {key}")
        images[key] = item.filename
    return images


def read_manifest(split):
    with (ROOT / "reference/splits" / f"{split}.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pairs(archive, split):
    if split == "tno49":
        data = read_manifest("tno49-origin-matches")
        if {row["image"] for row in data} != {row["image"] for row in read_manifest(split)}:
            raise ValueError("The source mapping does not cover the published TNO subset")
        prefix = "TNO_Image_Fusion_Dataset/"
        for row in data:
            shape = (int(row["height"]), int(row["width"]))
            yield row["image"], prefix + row["archive_ir"], prefix + row["archive_vi"], shape
        return

    infrared, visible, expected_count = LAYOUTS[split]
    ir = index(archive, infrared)
    vi = index(archive, visible)
    if set(ir) != set(vi) or len(ir) != expected_count:
        raise ValueError(f"{split} needs {expected_count} exact pairs, found {len(ir)} IR and {len(vi)} VI")
    if split != "msrs-train":
        paper_pairs = {Path(row["image"]).stem.casefold() for row in read_manifest(split)}
        if set(ir) != paper_pairs:
            raise ValueError(f"{split} stems differ from the paper evaluation manifest")
    for key in sorted(ir):
        yield None, ir[key], vi[key], None


def basename(member):
    return PurePosixPath(member.replace(chr(92), "/")).name


def decode(content, mode):
    return cv2.imdecode(np.frombuffer(content, dtype=np.uint8), mode)


def write_png(path, image):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"Could not write PNG for {path.name}")
    path.write_bytes(encoded.tobytes())


def prepare(archive, split, output):
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with TemporaryDirectory(prefix="prepare-", dir=output.parent) as temporary:
        staged = Path(temporary) / "images"
        for name in ("ir", "vi", "vi_y"):
            (staged / name).mkdir(parents=True)
        for alias, ir_member, vi_member, expected_shape in pairs(archive, split):
            ir_bytes, vi_bytes = archive.read(ir_member), archive.read(vi_member)
            ir = decode(ir_bytes, cv2.IMREAD_GRAYSCALE)
            luminance = decode(vi_bytes, cv2.IMREAD_GRAYSCALE)
            if ir is None or luminance is None or luminance.shape != ir.shape:
                raise ValueError(f"{split}: unreadable or unaligned pair {basename(ir_member)}")
            if expected_shape is not None and ir.shape != expected_shape:
                raise ValueError(f"{alias}: dimensions differ from the TNO audit")

            if alias is None:
                (staged / "ir" / basename(ir_member)).write_bytes(ir_bytes)
                (staged / "vi" / basename(vi_member)).write_bytes(vi_bytes)
                alias = Path(basename(ir_member)).stem + ".png"
            else:
                write_png(staged / "ir" / alias, ir)
                write_png(staged / "vi" / alias, decode(vi_bytes, cv2.IMREAD_UNCHANGED))
            write_png(staged / "vi_y" / alias, luminance)
            count += 1
        staged.rename(output)
    print(f"Prepared {count} aligned {split} pairs in {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--split", choices=(*LAYOUTS, "tno49"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with ZipFile(args.archive) as archive:
        prepare(archive, args.split, args.output_dir)


if __name__ == "__main__":
    main()
