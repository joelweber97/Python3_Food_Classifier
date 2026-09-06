# Python3 Food Classifier

Classifies a photo of food into one of the 101 [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
categories by fine-tuning an ImageNet-pretrained backbone.

**Best result: 92.6% top-1 / 98.8% top-5** on the official 25,250-image test
split, from a two-model ensemble with test-time augmentation.

| model | res | top-1 | top-5 |
| --- | --- | --- | --- |
| **ensemble (V2-L + V2-M) + TTA** | 300 | **92.60%** | **98.76%** |
| EfficientNetV2-L + TTA | 300 | 91.91% | 98.40% |
| EfficientNetV2-L | 300 | 91.68% | 98.32% |
| EfficientNetV2-M + TTA | 300 | 91.56% | 98.36% |
| EfficientNetV2-M | 300 | 91.25% | 98.30% |
| EfficientNetV2-B0 (first baseline) | 224 | 86.51% | 97.18% |

Reproduce the best number with:

```bash
python evaluate.py models/effnetv2l_300 models/effnetv2m_300 --tta
```

## Layout

| Path | What it is |
| --- | --- |
| `images/<class>/*.jpg` | the dataset - 101 classes x 1000 images |
| `meta/meta/` | official train/test split and class list |
| `train.py` | training (writes to `models/`) |
| `predict.py` | classify new images |
| `evaluate.py` | score models on the test split; ensembles and TTA |
| `models/` | trained models, weights, class lists, metrics (git-ignored) |

## Setup

Everything runs in the `tfgpu` conda environment (TensorFlow 2.9, CUDA):

```bash
conda activate tfgpu
```

## Training

```bash
python train.py --arch effnetv2l --img-size 300 --batch-size 24
python train.py --list-archs      # every supported backbone
```

Training runs in two phases - the classifier head with the backbone frozen, then
the whole backbone at a low, cosine-decayed learning rate. The best epoch by
validation accuracy is checkpointed, so an interrupted run still leaves a usable
model, and `--resume` continues from it.

Useful flags:

```bash
python train.py --arch effnetv2s --classes 25    # quick screen on 25 classes
python train.py --batch-size 16                  # if the GPU runs out of memory
python train.py --init-weights models/effnetv2l_300/best_weights.h5 \
                --img-size 384 --head-epochs 0   # progressive resizing
```

## Classifying an image

```bash
python predict.py photo.jpg --model models/effnetv2l_300
python predict.py photos/ --top 3
python predict.py photo.jpg --tta                        # mirror-average
python predict.py photo.jpg --model models/a models/b     # ensemble
```

Output is the top-k guesses with confidences:

```
photo.jpg
  1. baklava                       96.47%
  2. apple pie                      1.55%
  3. baby back ribs                 1.02%
```

Photos straight from a phone work: EXIF orientation is honoured and any common
image format is accepted.

## Backbone comparison

Every candidate was trained with an identical recipe on the same 25-class subset
(2 head + 5 fine-tune epochs) purely to rank them; the numbers are subset scores,
not final accuracy.

| arch | res | top-1 | top-5 | time |
| --- | --- | --- | --- | --- |
| effnetv2l | 300 | **93.41%** | 99.33% | 21.5 min |
| effnetv2m | 300 | 92.61% | 99.15% | 12.8 min |
| resnetrs152 | 256 | 91.87% | 99.15% | 9.9 min |
| effnetv2s | 300 | 91.71% | 99.10% | 7.4 min |
| resnetrs101 | 224 | 90.88% | 99.02% | 14.0 min |
| effnetv2b3 | 300 | 90.37% | 98.98% | 6.4 min |
| inceptionresnetv2 | 299 | 89.63% | 98.69% | 7.7 min |
| densenet201 | 224 | 89.12% | 98.77% | 5.2 min |
| effnetv2b2 | 260 | 89.09% | 98.69% | 3.4 min |
| xception | 299 | 87.20% | 98.11% | 4.7 min |
| resnet152v2 | 224 | 84.67% | 97.71% | 5.6 min |

EfficientNetV2 wins at every size, scaling cleanly B2 -> B3 -> S -> M -> L.
Parameter count alone does not predict quality: ResNetRS101 has 3x the
parameters of EfficientNetV2-S and still scores lower.

### What helped, and what did not

- **Backbone choice** was worth the most: +5.2 points, B0 -> V2-L.
- **Ensembling** V2-L with V2-M added +0.7 over the best single model. The two
  score within 0.4 of each other but make different mistakes, which is exactly
  the condition where averaging pays.
- **TTA** (averaging a prediction with its mirror image) added +0.2 for a 2x
  inference cost. Cheap, and it never hurt.
- **Progressive resizing to 384 did not work here** and was abandoned. Starting
  from the 91.7% model at 300px, fine-tuning at 384 scored 90.6% - a point
  *worse*. The 300px model had already reached 99.8% training accuracy, so there
  was no signal left to learn from; re-fine-tuning only overfitted further. This
  technique needs a model that has not yet saturated.

The hardest classes are genuinely ambiguous rather than badly modelled: `steak`
(68%) is confused with `filet_mignon` (79%) and `pork_chop` (80%), which are
near-identical photographs of grilled meat. `edamame` and `oysters` reach 99.6%.

## Notes on this setup

Things specific to TensorFlow 2.9 on an Ada-generation GPU, learned the hard way
and encoded in the scripts:

- **Augmentation is in the `tf.data` pipeline, not Keras layers.** TF 2.9's
  `RandomRotation`/`RandomZoom` were ~7x slower than the entire backbone on this
  GPU. Moving to `tf.image` ops on CPU workers cut epoch time by 3.7x.
- **Backbones' built-in preprocessing is disabled**; it divides float16
  activations by float32 constants, a hard error under `mixed_float16`.
  `train.py` redoes the normalisation in float32 layers. Each family expects a
  *different* normalisation, and getting it wrong costs accuracy silently, so
  every entry in `ARCHS` was read from the Keras source.
- **No XLA.** `jit_compile=True` fatally crashes: the `ptxas` shipped with TF 2.9
  (CUDA 11.2) cannot target this card. The `ptxas exited with non-zero error
  code` warnings during training are the autotuner probing, and are harmless.
- **RegNet cannot be trained here at all.** Its grouped convolutions fail to
  trace under mixed precision, and in float32 they route through XLA and hit the
  same fatal ptxas bug. `train.py` refuses them with an explanation.
- Train and predict must resize identically (resize, then centre-crop);
  `preprocess()` in `train.py` and `load_image()` in `predict.py` are kept in
  step, and each model records its resolution in `config.json`.
