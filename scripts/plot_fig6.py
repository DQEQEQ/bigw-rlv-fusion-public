"""Rebuild the manuscript's qualitative comparison from three MSRS scenes."""

import argparse
from base64 import b64encode
from io import BytesIO
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCENES = {
    "00959N": (160, 115, 400, 295),
    "00111D": (10, 185, 250, 365),
    "00345D": (275, 135, 555, 345),
}
METHODS = ("ir", "vi", "baseline", "bigw", "rlv-only", "bigw-rlv")


def find_pair_image(folder, scene):
    candidates = [
        p for p in folder.iterdir()
        if p.is_file() and p.stem.casefold() == scene.casefold()
        and p.suffix.casefold() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    ]
    if len(candidates) != 1:
        raise ValueError(f"Expected one image for {scene} in {folder}, found {len(candidates)}")
    with Image.open(candidates[0]) as image:
        if image.size != (640, 480):
            raise ValueError(f"{candidates[0]} must be an aligned 640 x 480 image")
        return image.convert("RGB")


def encode(image):
    with BytesIO() as buffer:
        image.save(buffer, format="PNG")
        return "data:image/png;base64," + b64encode(buffer.getvalue()).decode("ascii")


def build(folders, output):
    template = (ROOT / "reference/fig6-layout.svg").read_text(encoding="utf-8")
    for index, (scene, region) in enumerate(SCENES.items()):
        for column, method in enumerate(METHODS):
            original = find_pair_image(folders[method], scene)
            slot = index * len(METHODS) + column + 1
            # Matplotlib's SVG template flips the embedded rasters vertically.
            for offset, image in ((0, original), (18, original.crop(region))):
                template = template.replace(
                    f"IMAGE_SLOT_{slot + offset:02d}", encode(image.transpose(Image.Transpose.FLIP_TOP_BOTTOM))
                )
    if "IMAGE_SLOT_" in template:
        raise RuntimeError("The figure layout contains an unfilled image slot")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(template)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for method in METHODS:
        parser.add_argument(f"--{method}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build({method: getattr(args, method.replace("-", "_")) for method in METHODS}, args.output)


if __name__ == "__main__":
    main()
