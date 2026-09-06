"""Train a Food-101 classifier with transfer learning.

Uses the official Food-101 train/test split shipped in meta/meta/ so results are
reproducible and comparable to published numbers.

Training runs in two phases:
  1. head-only  - backbone frozen, learns the 101-way classifier quickly
  2. fine-tune  - backbone unfrozen (BatchNorm kept frozen) at a low LR

Example:
    python train.py --arch effnetv2s              # best single-model config
    python train.py --arch effnetv2b0 --classes 5 # quick smoke test
    python train.py --list-archs
"""
import argparse
import json
import time
from pathlib import Path

import tensorflow as tf

ROOT = Path(__file__).resolve().parent
IMAGES_DIR = ROOT / "images"
META_DIR = ROOT / "meta" / "meta"

# Each backbone declares the input normalisation its ImageNet weights expect, the
# resolution those weights were trained at, and whether its Keras constructor
# accepts include_preprocessing. Getting the normalisation wrong silently costs
# several points of accuracy, so every entry is taken from the Keras source
# rather than assumed. The four conventions in play:
#   imagenet - x/255 then ImageNet mean/std   (Keras "torch" mode)
#   pm1_128  - x/128 - 1                      (EfficientNetV2 S/M/L only)
#   pm1      - x/127.5 - 1                    (Keras "tf" mode)
#   none     - raw [0,255]; the model rescales internally (RegNet)
# ResNet V1 and VGG are deliberately absent: they need "caffe" BGR preprocessing
# and are not competitive with anything below.
# "nested" backbones are built with input_shape and then called on the
# preprocessed tensor; RegNet's squeeze-and-excite blocks fail shape inference
# when built with input_tensor, so they have to be attached as a submodel.
# "broken" backbones cannot train on this TF 2.9 / Ada-GPU combination at all.
# The RegNets are all grouped-convolution networks, and that path fails twice
# over: under mixed_float16 TF cannot trace a grouped Conv2D whose kernel is an
# AutoCastVariable, and in float32 the grouped conv is compiled through XLA,
# whose bundled ptxas (CUDA 11.2) cannot emit SASS for this card and aborts the
# process. Fixing it means replacing the CUDA toolchain in the environment.
def _a(keras, norm, size, ip=False, nested=False, broken=None):
    return {"keras": keras, "norm": norm, "size": size, "ip": ip,
            "nested": nested, "broken": broken}


REGNET_BROKEN = ("RegNet needs grouped convolutions, which crash on TF 2.9 with "
                 "this GPU (ptxas cannot target Ada). No training flag fixes it.")


ARCHS = {
    "effnetv2b0": _a("EfficientNetV2B0", "imagenet", 224, ip=True),
    "effnetv2b1": _a("EfficientNetV2B1", "imagenet", 240, ip=True),
    "effnetv2b2": _a("EfficientNetV2B2", "imagenet", 260, ip=True),
    "effnetv2b3": _a("EfficientNetV2B3", "imagenet", 300, ip=True),
    "effnetv2s": _a("EfficientNetV2S", "pm1_128", 300, ip=True),
    "effnetv2m": _a("EfficientNetV2M", "pm1_128", 384, ip=True),
    "effnetv2l": _a("EfficientNetV2L", "pm1_128", 384, ip=True),
    "resnetrs50": _a("ResNetRS50", "imagenet", 224, ip=True),
    "resnetrs101": _a("ResNetRS101", "imagenet", 224, ip=True),
    "resnetrs152": _a("ResNetRS152", "imagenet", 256, ip=True),
    "resnetrs200": _a("ResNetRS200", "imagenet", 256, ip=True),
    "regnety032": _a("RegNetY032", "none", 224, nested=True, broken=REGNET_BROKEN),
    "regnety080": _a("RegNetY080", "none", 224, nested=True, broken=REGNET_BROKEN),
    "regnety160": _a("RegNetY160", "none", 224, nested=True, broken=REGNET_BROKEN),
    "regnety320": _a("RegNetY320", "none", 224, nested=True, broken=REGNET_BROKEN),
    "xception": _a("Xception", "pm1", 299),
    "inceptionresnetv2": _a("InceptionResNetV2", "pm1", 299),
    "inceptionv3": _a("InceptionV3", "pm1", 299),
    "resnet152v2": _a("ResNet152V2", "pm1", 224),
    "densenet201": _a("DenseNet201", "imagenet", 224),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", default="effnetv2b0", choices=sorted(ARCHS),
                   help="backbone to fine-tune")
    p.add_argument("--img-size", type=int, default=0,
                   help="input resolution (0 = the backbone's native size)")
    p.add_argument("--out", type=Path, default=None,
                   help="output directory (default: models/<arch>_<size>)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--head-epochs", type=int, default=5,
                   help="epochs with the backbone frozen")
    p.add_argument("--finetune-epochs", type=int, default=15,
                   help="epochs with the backbone unfrozen")
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--finetune-lr", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1,
                   help="Food-101 labels are noisy; smoothing helps")
    p.add_argument("--classes", type=int, default=0,
                   help="train on only the first N classes (0 = all 101)")
    p.add_argument("--init-weights", type=Path, default=None,
                   help="start from these weights, e.g. the same arch trained at "
                        "a lower resolution (progressive resizing)")
    p.add_argument("--no-mixed-precision", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="restore best weights from --out before training")
    p.add_argument("--list-archs", action="store_true")
    return p.parse_args()


def read_split(split, class_to_idx):
    """Read meta/meta/{train,test}.txt into (filepaths, labels)."""
    paths, labels = [], []
    missing = 0
    for line in (META_DIR / (split + ".txt")).read_text().split():
        cls, _, stem = line.partition("/")
        if cls not in class_to_idx:
            continue
        path = IMAGES_DIR / cls / (stem + ".jpg")
        if not path.exists():
            missing += 1
            continue
        paths.append(str(path))
        labels.append(class_to_idx[cls])
    if missing:
        print("  warning: {} file(s) listed in {}.txt are missing on disk".format(missing, split))
    return paths, labels


def preprocess(img, img_size, resize_to, training):
    """Decode-side image prep, shared by training, validation and predict.py.

    Augmentation lives here rather than in Keras layers on purpose: TF 2.9's
    RandomRotation/RandomZoom kernels are ~7x slower than the whole backbone on
    this GPU. These tf.image ops run on CPU workers overlapped with GPU compute,
    so they are effectively free.
    """
    img = tf.image.resize(img, [resize_to, resize_to])
    if training:
        img = tf.image.random_crop(img, [img_size, img_size, 3])
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 24.0)
        img = tf.image.random_contrast(img, 0.85, 1.15)
        img = tf.image.random_saturation(img, 0.85, 1.15)
        img = tf.clip_by_value(img, 0.0, 255.0)
    else:
        # Centre crop, so validation sees the same framing as a training crop.
        off = (resize_to - img_size) // 2
        img = tf.image.crop_to_bounding_box(img, off, off, img_size, img_size)
    return img


def build_dataset(paths, labels, num_classes, batch_size, img_size, resize_to, training):
    """File paths -> batched (image, one-hot) dataset."""
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        # Shuffle the full file list every epoch; cheap, since these are just paths.
        ds = ds.shuffle(len(paths), reshuffle_each_iteration=True)

    def load(path, label):
        img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
        # Pixels stay in [0, 255]; the model normalises them internally.
        return (preprocess(img, img_size, resize_to, training),
                tf.one_hot(label, num_classes))

    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model(arch, img_size, num_classes):
    """Backbone that takes raw [0,255] pixels and returns class probabilities.

    Normalisation is part of the model so predict.py only has to resize and crop.

    The backbone's built-in preprocessing is disabled (include_preprocessing=
    False) because it divides float16 activations by float32 constants, which is
    a hard error under a mixed_float16 policy. Doing the same normalisation in
    float32 layers here sidesteps that; the first conv casts back to float16.
    """
    spec = ARCHS[arch]
    norm = spec["norm"]
    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="image")
    if norm == "imagenet":
        x = tf.keras.layers.Rescaling(1.0 / 255, dtype="float32")(inputs)
        x = tf.keras.layers.Normalization(
            mean=[0.485, 0.456, 0.406],
            variance=[0.229 ** 2, 0.224 ** 2, 0.225 ** 2],
            axis=-1, dtype="float32")(x)
    elif norm == "pm1_128":
        x = tf.keras.layers.Rescaling(1.0 / 128.0, offset=-1, dtype="float32")(inputs)
    elif norm == "pm1":
        x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1, dtype="float32")(inputs)
    else:  # "none" - the backbone rescales internally
        x = tf.keras.layers.Activation("linear", dtype="float32")(inputs)

    ctor = getattr(tf.keras.applications, spec["keras"])
    kwargs = {"include_preprocessing": False} if spec["ip"] else {}
    if spec["nested"]:
        base = ctor(include_top=False, weights="imagenet",
                    input_shape=(img_size, img_size, 3), **kwargs)
        features = base(x)
    else:
        base = ctor(include_top=False, weights="imagenet", input_tensor=x, **kwargs)
        features = base.output
    base.trainable = False

    x = tf.keras.layers.GlobalAveragePooling2D(name="pool")(features)
    x = tf.keras.layers.BatchNormalization(name="pool_bn")(x)
    x = tf.keras.layers.Dropout(0.3, name="top_dropout")(x)
    # float32 on the last layer keeps the softmax numerically safe under mixed precision.
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax",
                                    dtype="float32", name="predictions")(x)
    return tf.keras.Model(inputs, outputs, name="food_classifier"), base


def compile_model(model, lr, label_smoothing):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5"),
        ],
    )


def main():
    args = parse_args()
    if args.list_archs:
        print("{:<20} {:<22} {:<10} {}".format("name", "keras model", "norm", "native size"))
        for name, s in sorted(ARCHS.items()):
            print("{:<20} {:<22} {:<10} {}".format(
                name, s["keras"], s["norm"], s["size"]))
        return

    if ARCHS[args.arch]["broken"]:
        raise SystemExit("{} is not usable here: {}".format(
            args.arch, ARCHS[args.arch]["broken"]))

    img_size = args.img_size or ARCHS[args.arch]["size"]
    # Resize slightly larger than the crop so random cropping has room to move.
    resize_to = int(round(img_size * 256 / 224))
    out = args.out or ROOT / "models" / "{}_{}".format(args.arch, img_size)
    out.mkdir(parents=True, exist_ok=True)

    gpus = tf.config.list_physical_devices("GPU")
    print("TensorFlow {} | GPUs: {}".format(
        tf.__version__, [g.name for g in gpus] or "NONE - running on CPU"))
    for gpu in gpus:
        # Grow memory on demand so other processes can still use the card.
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus and not args.no_mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("mixed precision: mixed_float16")
    print("arch={} img_size={} resize_to={} batch_size={}".format(
        args.arch, img_size, resize_to, args.batch_size))

    class_names = (META_DIR / "classes.txt").read_text().split()
    if args.classes:
        class_names = class_names[:args.classes]
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    num_classes = len(class_names)

    train_paths, train_labels = read_split("train", class_to_idx)
    val_paths, val_labels = read_split("test", class_to_idx)
    print("{} classes | {} train | {} val".format(
        num_classes, len(train_paths), len(val_paths)))

    (out / "class_names.json").write_text(json.dumps(class_names, indent=2))
    # predict.py reads this so inference matches training framing exactly.
    (out / "config.json").write_text(json.dumps(
        {"arch": args.arch, "img_size": img_size, "resize_to": resize_to}, indent=2))

    train_ds = build_dataset(train_paths, train_labels, num_classes,
                             args.batch_size, img_size, resize_to, True)
    val_ds = build_dataset(val_paths, val_labels, num_classes,
                           args.batch_size, img_size, resize_to, False)

    model, base = build_model(args.arch, img_size, num_classes)
    weights_path = out / "best_weights.h5"
    if args.resume and weights_path.exists():
        model.load_weights(str(weights_path))
        print("resumed weights from {}".format(weights_path))
    elif args.init_weights:
        # Convolutional weights are resolution-independent, so a model trained at
        # a smaller size is a much better starting point than ImageNet alone.
        model.load_weights(str(args.init_weights))
        print("initialised from {}".format(args.init_weights))

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(weights_path), monitor="val_accuracy", mode="max",
            save_best_only=True, save_weights_only=True, verbose=1),
        tf.keras.callbacks.CSVLogger(str(out / "history.csv"), append=True),
    ]

    t0 = time.time()
    if args.head_epochs:
        print("\n=== phase 1/2: training classifier head (backbone frozen) ===")
        compile_model(model, args.head_lr, args.label_smoothing)
        model.fit(train_ds, validation_data=val_ds, epochs=args.head_epochs,
                  callbacks=callbacks, verbose=2)

    if args.finetune_epochs:
        print("\n=== phase 2/2: fine-tuning backbone ===")
        base.trainable = True
        # BatchNorm stays frozen: batch statistics from fine-tuning would wreck
        # the pretrained running statistics.
        frozen_bn = 0
        for layer in base.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
                frozen_bn += 1
        print("unfroze backbone, kept {} BatchNorm layers frozen".format(frozen_bn))

        steps = max(1, len(train_paths) // args.batch_size) * args.finetune_epochs
        schedule = tf.keras.optimizers.schedules.CosineDecay(args.finetune_lr, steps)
        compile_model(model, schedule, args.label_smoothing)
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max", patience=4,
            restore_best_weights=True, verbose=1))
        model.fit(train_ds, validation_data=val_ds, epochs=args.finetune_epochs,
                  callbacks=callbacks, verbose=2)

    if weights_path.exists():
        model.load_weights(str(weights_path))
    loss, acc, top5 = model.evaluate(val_ds, verbose=0)
    mins = (time.time() - t0) / 60
    print("\nbest model: val_accuracy={:.4f}  val_top5={:.4f}  loss={:.4f}".format(acc, top5, loss))
    print("total training time: {:.1f} min".format(mins))

    saved = out / "saved_model"
    model.save(str(saved))
    print("saved model -> {}".format(saved))
    (out / "metrics.json").write_text(json.dumps(
        {"arch": args.arch, "img_size": img_size,
         "val_accuracy": float(acc), "val_top5_accuracy": float(top5),
         "val_loss": float(loss), "num_classes": num_classes,
         "train_images": len(train_paths), "val_images": len(val_paths),
         "training_minutes": round(mins, 1)}, indent=2))


if __name__ == "__main__":
    main()
