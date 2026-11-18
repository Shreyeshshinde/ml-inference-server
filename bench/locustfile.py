"""Locust load test: every simulated user uploads images back to back (closed loop).

Environment variables:
  MODEL       model name in the URL (default resnet50)
  IMAGE_DIR   Imagenette val folder (default data/imagenette2-320/val)
  NUM_IMAGES  how many distinct images to cycle through (default 50)

Run headless, e.g.:
  MODEL=resnet50 locust -f bench/locustfile.py --headless -u 100 -r 100 -t 60s \
      --host http://localhost:8080 --csv results/raw/run
Any status other than 200 is a failure, recorded as "HTTP <code>" so 503s are countable.
"""
import os
import random
from pathlib import Path

from locust import FastHttpUser, constant, task

MODEL = os.environ.get("MODEL", "resnet50")
IMAGE_DIR = Path(os.environ.get("IMAGE_DIR", "data/imagenette2-320/val"))
NUM_IMAGES = int(os.environ.get("NUM_IMAGES", "50"))
BOUNDARY = "----mis-bench-boundary"


def multipart_body(image_bytes: bytes) -> bytes:
    """Builds the multipart/form-data body by hand, once per image, so the load
    generator spends its CPU sending requests, not encoding them."""
    head = (f"--{BOUNDARY}\r\n"
            'Content-Disposition: form-data; name="image"; filename="img.jpg"\r\n'
            "Content-Type: image/jpeg\r\n\r\n").encode()
    return head + image_bytes + f"\r\n--{BOUNDARY}--\r\n".encode()


def load_bodies() -> list[bytes]:
    # Same deterministic image set for every run and both servers: an equal share
    # from each class, sorted by file name.
    class_dirs = sorted(p for p in IMAGE_DIR.iterdir() if p.is_dir())
    per_class = max(1, NUM_IMAGES // len(class_dirs))
    files = [f for d in class_dirs for f in sorted(d.glob("*.JPEG"))[:per_class]]
    return [multipart_body(f.read_bytes()) for f in files[:NUM_IMAGES]]


BODIES = load_bodies()
HEADERS = {"Content-Type": f"multipart/form-data; boundary={BOUNDARY}"}


class ImageUser(FastHttpUser):
    wait_time = constant(0)  # closed loop: next request as soon as the last one returns

    @task
    def predict(self):
        with self.client.post(f"/v1/models/{MODEL}/predict", data=random.choice(BODIES),
                              headers=HEADERS, name=MODEL, catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
