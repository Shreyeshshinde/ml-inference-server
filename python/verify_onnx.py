"""Check that the exported ONNX models compute the same thing as PyTorch.

Runs 20 Imagenette val images through both, as one batch (which also tests the
dynamic batch axis), and asserts max |difference| of logits < 1e-4.
"""
import numpy as np
import onnxruntime as ort
import torch
import torchvision

from common import MODEL_NAMES, MODELS_DIR, load_image, preprocess_bgr, sample_val_images

TOLERANCE = 1e-4


def main() -> None:
    images = sample_val_images(20)
    batch = np.stack([preprocess_bgr(load_image(p)) for p in images])  # [20, 3, 224, 224]

    failed = False
    for name in MODEL_NAMES:
        model = getattr(torchvision.models, name)(weights="IMAGENET1K_V2").eval()
        with torch.no_grad():
            torch_out = model(torch.from_numpy(batch)).numpy()

        sess = ort.InferenceSession(str(MODELS_DIR / f"{name}.onnx"),
                                    providers=["CPUExecutionProvider"])
        ort_out = sess.run(["logits"], {"input": batch})[0]

        max_diff = float(np.abs(torch_out - ort_out).max())
        same_top1 = int((torch_out.argmax(1) == ort_out.argmax(1)).sum())
        status = "PASS" if max_diff < TOLERANCE else "FAIL"
        failed |= status == "FAIL"
        print(f"{name:20s} max_abs_diff={max_diff:.2e}  top1_agree={same_top1}/{len(images)}  {status}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
