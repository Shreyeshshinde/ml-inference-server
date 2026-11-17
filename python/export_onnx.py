"""Export torchvision ImageNet classifiers to ONNX with a dynamic batch axis.

Usage: python python/export_onnx.py [--models resnet50 mobilenet_v3_large]
Writes models/<name>.onnx and data/imagenet_labels.txt.
"""
import argparse

import onnx
import onnxsim
import torch
import torchvision

from common import LABELS_PATH, MODEL_NAMES, MODELS_DIR

WEIGHTS = {
    "resnet50": torchvision.models.ResNet50_Weights.IMAGENET1K_V2,
    "mobilenet_v3_large": torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V2,
}


def export(name: str) -> None:
    weights = WEIGHTS[name]
    model = getattr(torchvision.models, name)(weights=weights).eval()
    dummy = torch.randn(1, 3, 224, 224)
    out_path = MODELS_DIR / f"{name}.onnx"

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["logits"],
        # Batch dimension is symbolic, so the same file serves batch 1, 8, 16...
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )

    # onnxsim folds constants and removes redundant nodes.
    # check_n=3 runs 3 random inputs through both graphs to confirm outputs still match.
    simplified, ok = onnxsim.simplify(onnx.load(str(out_path)), check_n=3)
    if not ok:
        raise RuntimeError(f"onnxsim could not validate simplified {name}")
    onnx.save(simplified, str(out_path))
    onnx.checker.check_model(str(out_path))
    print(f"exported {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)
    for name in args.models:
        export(name)

    # Class names come from torchvision metadata, one per line, line index = class id.
    categories = WEIGHTS["resnet50"].meta["categories"]
    assert len(categories) == 1000
    LABELS_PATH.write_text("\n".join(categories) + "\n")
    print(f"wrote {LABELS_PATH}")


if __name__ == "__main__":
    main()
