"""Imagenette helpers: class mapping to ImageNet indices and deterministic image lists."""
from pathlib import Path

from common import IMAGENETTE_DIR, LABELS_PATH

# Imagenette folder name (WordNet id) -> ImageNet-1k class index.
# The expected label text is checked against data/imagenet_labels.txt at runtime,
# so a wrong index fails loudly instead of silently producing a bad accuracy number.
WNID_TO_IMAGENET = {
    "n01440764": (0, "tench"),
    "n02102040": (217, "English springer"),
    "n02979186": (482, "cassette player"),
    "n03000684": (491, "chain saw"),
    "n03028079": (497, "church"),
    "n03394916": (566, "French horn"),
    "n03417042": (569, "garbage truck"),
    "n03425413": (571, "gas pump"),
    "n03445777": (574, "golf ball"),
    "n03888257": (701, "parachute"),
}


def check_mapping() -> None:
    labels = LABELS_PATH.read_text().splitlines()
    for wnid, (idx, name) in WNID_TO_IMAGENET.items():
        if labels[idx] != name:
            raise SystemExit(f"class mapping error: {wnid} -> {idx} is '{labels[idx]}', "
                             f"expected '{name}'")


def list_split(split: str, per_class: int | None = None) -> list[tuple[Path, int]]:
    """(image path, ImageNet class index) pairs, sorted for reproducibility.

    per_class limits how many images are taken from each class (None = all).
    """
    split_dir = IMAGENETTE_DIR / split
    if not split_dir.exists():
        raise SystemExit("Imagenette missing. Run: bash scripts/download_data.sh")
    items = []
    for wnid, (idx, _) in sorted(WNID_TO_IMAGENET.items()):
        files = sorted((split_dir / wnid).glob("*.JPEG"))
        if per_class is not None:
            files = files[:per_class]
        items.extend((f, idx) for f in files)
    return items
