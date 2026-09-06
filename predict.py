"""Classify food images with a model trained by train.py.

    python predict.py photo.jpg
    python predict.py photo1.jpg photo2.png --top 3
    python predict.py some_folder/                     # every image in the folder
    python predict.py photo.jpg --tta                  # slower, slightly better
    python predict.py photo.jpg --model models/a models/b   # ensemble
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", type=Path,
                   help="image files, or folders of images, to classify")
    p.add_argument("--model", nargs="+", type=Path, default=None,
                   help="model directory produced by train.py; pass several to "
                        "average their predictions (default: the best-scoring "
                        "model in models/)")
    p.add_argument("--top", type=int, default=5, help="how many guesses to show")
    p.add_argument("--tta", action="store_true",
                   help="also predict on the mirrored image and average")
    return p.parse_args()


def best_model_dir():
    """The trained model with the highest recorded validation accuracy.

    Saves having to remember which run won; `python predict.py photo.jpg` just
    uses the best model available.
    """
    candidates = []
    for metrics in (ROOT / "models").glob("*/metrics.json"):
        if not (metrics.parent / "saved_model").is_dir():
            continue
        try:
            candidates.append((json.loads(metrics.read_text())["val_accuracy"],
                               metrics.parent))
        except (ValueError, KeyError):
            continue
    if not candidates:
        raise SystemExit("no trained models in models/ - run train.py first, "
                         "or pass --model")
    return max(candidates)[1]


def collect_images(paths):
    files = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(f for f in path.iterdir()
                                if f.suffix.lower() in SUFFIXES))
        elif path.is_file():
            files.append(path)
        else:
            print("skipping {} - not found".format(path))
    return files


def load_image(path, img_size, resize_to):
    """Load any image as an img_size square RGB array in [0, 255].

    Mirrors the validation path in train.py exactly - resize then centre crop -
    because a model is only as good as the framing it was tuned on.

    exif_transpose matters for phone photos: they are often stored sideways with
    an orientation tag that only some readers honour.
    """
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    img = img.resize((resize_to, resize_to), Image.BILINEAR)
    off = (resize_to - img_size) // 2
    img = img.crop((off, off, off + img_size, off + img_size))
    return np.asarray(img, dtype=np.float32)


def main():
    args = parse_args()
    files = collect_images(args.paths)
    if not files:
        print("no images to classify")
        return
    if args.model is None:
        args.model = [best_model_dir()]
        print("using {}".format(args.model[0]))

    import tensorflow as tf  # imported late so --help stays instant

    class_names = None
    probs = None
    for model_dir in args.model:
        cfg = json.loads((model_dir / "config.json").read_text())
        names = json.loads((model_dir / "class_names.json").read_text())
        if class_names is None:
            class_names = names
        elif names != class_names:
            raise SystemExit("models disagree on class list; cannot ensemble")

        model = tf.keras.models.load_model(str(model_dir / "saved_model"))
        batch = np.stack([load_image(f, cfg["img_size"], cfg["resize_to"])
                          for f in files])
        p = model.predict(batch, verbose=0)
        if args.tta:
            p = (p + model.predict(batch[:, :, ::-1, :], verbose=0)) / 2
        probs = p if probs is None else probs + p

    probs /= len(args.model)
    top_n = min(args.top, len(class_names))
    for path, row in zip(files, probs):
        print("\n{}".format(path))
        for rank, idx in enumerate(np.argsort(row)[::-1][:top_n], start=1):
            label = class_names[idx].replace("_", " ")
            print("  {}. {:<28} {:6.2f}%".format(rank, label, 100 * row[idx]))


if __name__ == "__main__":
    main()
