"""Evaluate trained models on the official Food-101 test split.

    python evaluate.py models/effnetv2s_300
    python evaluate.py models/effnetv2s_300 --tta
    python evaluate.py models/a models/b --tta      # ensemble the two
    python evaluate.py models/effnetv2s_300 --per-class

Ensembling averages the models' probabilities, so each model may run at its own
input resolution.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
IMAGES_DIR = ROOT / "images"
META_DIR = ROOT / "meta" / "meta"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="+", type=Path,
                   help="model directories produced by train.py")
    p.add_argument("--tta", action="store_true",
                   help="average each model with its mirrored prediction")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--per-class", action="store_true",
                   help="list the worst and best classes")
    return p.parse_args()


def main():
    args = parse_args()
    import tensorflow as tf
    from train import preprocess

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    # No global mixed-precision policy here: a SavedModel restores the dtype of
    # each layer from its own config, and forcing float16 breaks the backbones
    # that can only run in float32 (RegNet).

    class_names = json.loads((args.models[0] / "class_names.json").read_text())
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    paths, labels = [], []
    for line in (META_DIR / "test.txt").read_text().split():
        cls, _, stem = line.partition("/")
        if cls not in class_to_idx:
            continue
        path = IMAGES_DIR / cls / (stem + ".jpg")
        if path.exists():
            paths.append(str(path))
            labels.append(class_to_idx[cls])
    labels = np.array(labels)
    print("{} test images | {} classes".format(len(paths), len(class_names)))

    total = None
    for model_dir in args.models:
        cfg = json.loads((model_dir / "config.json").read_text())
        names = json.loads((model_dir / "class_names.json").read_text())
        if names != class_names:
            raise SystemExit("models disagree on class list; cannot ensemble")

        def load(path):
            img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
            return preprocess(img, cfg["img_size"], cfg["resize_to"], False)

        ds = (tf.data.Dataset.from_tensor_slices(paths)
              .map(load, num_parallel_calls=tf.data.AUTOTUNE)
              .batch(args.batch_size).prefetch(tf.data.AUTOTUNE))

        model = tf.keras.models.load_model(str(model_dir / "saved_model"))
        probs = model.predict(ds, verbose=0)
        if args.tta:
            flipped = ds.map(lambda x: x[:, :, ::-1, :],
                             num_parallel_calls=tf.data.AUTOTUNE)
            probs = (probs + model.predict(flipped, verbose=0)) / 2

        top1 = (probs.argmax(1) == labels).mean()
        top5 = np.mean([labels[i] in np.argpartition(probs[i], -5)[-5:]
                        for i in range(len(labels))])
        print("  {:<28} arch={:<12} size={:<4} top1={:.4f}  top5={:.4f}".format(
            model_dir.name, cfg["arch"], cfg["img_size"], top1, top5))
        total = probs if total is None else total + probs
        tf.keras.backend.clear_session()

    if len(args.models) > 1:
        total /= len(args.models)
        top1 = (total.argmax(1) == labels).mean()
        top5 = np.mean([labels[i] in np.argpartition(total[i], -5)[-5:]
                        for i in range(len(labels))])
        print("  {:<28} {:>31} top1={:.4f}  top5={:.4f}".format(
            "ENSEMBLE of {}".format(len(args.models)), "", top1, top5))

    if args.per_class:
        preds = total.argmax(1)
        correct, count = defaultdict(int), defaultdict(int)
        for label, pred in zip(labels, preds):
            count[label] += 1
            correct[label] += int(label == pred)
        acc = sorted((correct[i] / count[i], class_names[i]) for i in count)
        print("\nworst 10 classes:")
        for a, name in acc[:10]:
            print("  {:<28} {:.1%}".format(name, a))
        print("best 10 classes:")
        for a, name in acc[-10:][::-1]:
            print("  {:<28} {:.1%}".format(name, a))


if __name__ == "__main__":
    main()
