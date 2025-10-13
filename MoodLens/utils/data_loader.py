# data_loader.py
"""
Improved data loading utilities for FER-2013 (7 classes) using tf.data.
Key changes vs. original:
- Use `image_dataset_from_directory` with *consistent* validation_split/subset to avoid data leakage.
- Enforce RGB and 224x224 to match VGG16 pretrained ImageNet expectations.
- Add class-weights computation from the training split.
- Provide a `prepare_datasets` function returning (train, val, test) datasets *and* class weights.
- Use AUTOTUNE and cache/prefetch to speed up pipeline.
- Keep augmentations outside this file (done in the model script) so this module focuses on I/O.
"""

import tensorflow as tf
from collections import Counter
from typing import Tuple, Dict

AUTOTUNE = tf.data.AUTOTUNE

# Expected FER-2013 7-class names (directory names)
FER7_CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

def _check_class_names(ds: tf.data.Dataset):
    """Optional: sanity check that directory class names match expected FER-2013 labels."""
    class_names = ds.class_names
    missing = [c for c in FER7_CLASSES if c not in class_names]
    if missing:
        tf.print("[warn] Some expected classes are missing in the dataset directories:", missing)
    return class_names

def prepare_datasets(
    train_dir: str = "data/train",
    test_dir: str = "data/test",
    img_size: Tuple[int, int] = (224, 224),
    batch_size: int = 64,
    val_split: float = 0.2,
    seed: int = 42,
):
    """
    Creates train/val/test datasets *without leakage* using image_dataset_from_directory.
    - Train/Val come from the SAME directory (train_dir) via validation_split/subset.
    - Test comes from a separate directory (test_dir) with no split.
    - color_mode='rgb' and target_size=224x224 to align with VGG16 pretraining.
    Returns:
        train_ds, val_ds, test_ds, class_names, class_weights (dict index->weight)
    """
    # Train subset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        color_mode="rgb",
        shuffle=True,
    )

    # Validation subset (from the same directory, same split & seed)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        color_mode="rgb",
        shuffle=False,
    )

    # Optional: sanity check class names
    class_names = _check_class_names(train_ds)

    # Test dataset (separate directory)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=img_size,
        batch_size=batch_size,
        color_mode="rgb",
        shuffle=False,
    )

    # Improve I/O throughput
    def _configure(ds, training: bool):
        ds = ds.cache() if not training else ds  # cache only non-augmented pipelines
        return ds.prefetch(AUTOTUNE)

    train_ds = _configure(train_ds, training=True)
    val_ds = _configure(val_ds, training=False)
    test_ds = _configure(test_ds, training=False)

    # Compute class weights from the *training* subset only (to fight class imbalance)
    class_weights = compute_class_weights_from_dataset(train_ds, len(class_names))

    return train_ds, val_ds, test_ds, class_names, class_weights


def compute_class_weights_from_dataset(train_ds: tf.data.Dataset, num_classes: int) -> Dict[int, float]:
    """
    Compute class weights for imbalanced data:
        weight_c = N_total / (num_classes * N_c)
    where N_c is count for class c.
    """
    counts = Counter()
    total = 0
    for _, y in train_ds.unbatch():
        # y is one-hot vector
        cls = int(tf.argmax(y).numpy())
        counts[cls] += 1
        total += 1

    if total == 0:
        return {i: 1.0 for i in range(num_classes)}

    weights = {}
    for i in range(num_classes):
        n_c = counts.get(i, 1)
        weights[i] = total / (num_classes * n_c)
    return weights
