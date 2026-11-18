"""INT8 static quantization of the exported ONNX models.

Usage: python python/quantize.py [--models resnet50 mobilenet_v3_large] [--method MinMax]
Writes models/<name>_int8.onnx.

- Static: activation ranges are measured once on calibration images and stored in
  the model, so inference does not have to measure them per request.
- Calibration uses 200 images from the Imagenette TRAIN split (20 per class).
  Validation images are never seen here, so the accuracy eval stays unbiased.
- QDQ format + per-channel weights (one scale per conv output channel).
"""
import argparse

import numpy as np
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat,
                                      QuantType, quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

from common import MODEL_NAMES, MODELS_DIR, load_image, preprocess_bgr
from imagenette import check_mapping, list_split

CALIB_PER_CLASS = 20  # x 10 classes = 200 images

# Per-model recipe (calibration method, op types to quantize; None = all).
# Chosen by measurement on Imagenette val (results/mobilenet_int8_experiments.csv):
# - resnet50: default MinMax on all ops already keeps top-1 within 0.13 points.
# - mobilenet_v3_large: MinMax on all ops lost 13 points of top-1 AND ran 3.7x slower
#   (many quantize/dequantize steps around HardSwish / squeeze-excite Mul, which have
#   no fast INT8 kernel). Quantizing only Conv/Gemm fixed the speed; Percentile
#   calibration (ignores the extreme 0.01% of activations) fixed most of the accuracy.
RECIPES = {
    "resnet50": ("MinMax", None),
    "mobilenet_v3_large": ("Percentile", ["Conv", "Gemm"]),
}


class ImagenetteCalibrationReader(CalibrationDataReader):
    """Feeds preprocessed train images to the calibrator, one batch at a time."""

    def __init__(self, input_name: str, batch_size: int = 10):
        paths = [p for p, _ in list_split("train", per_class=CALIB_PER_CLASS)]
        assert len(paths) == 10 * CALIB_PER_CLASS
        # Same preprocessing as serving, so calibrated ranges match real traffic.
        data = np.stack([preprocess_bgr(load_image(p)) for p in paths])
        self.batches = iter([{input_name: data[i:i + batch_size]}
                             for i in range(0, len(data), batch_size)])

    def get_next(self):
        return next(self.batches, None)  # None tells ORT calibration is finished


def quantize(name: str, method: str, op_types: list[str] | None = None,
             suffix: str = "_int8") -> None:
    fp32 = MODELS_DIR / f"{name}.onnx"
    prepped = MODELS_DIR / f"{name}_prep.onnx"
    int8 = MODELS_DIR / f"{name}{suffix}.onnx"

    # Recommended pre-step: symbolic shape inference + graph cleanup, so the quantizer
    # knows every tensor's shape and can quantize more ops.
    quant_pre_process(str(fp32), str(prepped))

    quantize_static(
        model_input=str(prepped),
        model_output=str(int8),
        calibration_data_reader=ImagenetteCalibrationReader("input"),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=getattr(CalibrationMethod, method),
        # None = quantize every op type ORT supports. A list (e.g. Conv, Gemm) keeps
        # everything else in FP32, which avoids extra quantize/dequantize steps around
        # ops that have no fast INT8 kernel.
        op_types_to_quantize=op_types,
    )
    prepped.unlink()
    print(f"{name}: {fp32.stat().st_size / 1e6:.1f} MB -> {int8.stat().st_size / 1e6:.1f} MB "
          f"({int8.name}, calibration={method}, ops={op_types or 'all'})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    parser.add_argument("--method", default=None, choices=["MinMax", "Entropy", "Percentile"],
                        help="override the per-model recipe's calibration method")
    parser.add_argument("--op-types", nargs="+", default=None,
                        help="override the recipe: only quantize these op types, e.g. Conv Gemm")
    parser.add_argument("--suffix", default="_int8", help="output file suffix")
    args = parser.parse_args()
    check_mapping()
    for name in args.models:
        method, op_types = RECIPES[name]
        quantize(name, args.method or method, args.op_types or op_types, args.suffix)


if __name__ == "__main__":
    main()
