# train_vgg16.py
"""
VGG16 fine-tuning for FER-2013 with a *correct* pipeline and strong baselines.

Why these changes matter:
- Input 224x224 RGB + VGG16 preprocess_input: matches pretraining statistics (ImageNet) → better transfer.
- Strict non-leaking split: train/val come from the same directory with validation_split; test is separate.
- Two-phase training: (1) freeze backbone to warm up the new head, (2) unfreeze top blocks and fine-tune with low LR.
- Class weights: counter class imbalance common in FER (e.g., 'disgust' usually rare).
- Better metrics: keep accuracy but also compute macro-F1 and confusion matrix after training.
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint for more stable convergence.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # quieter logs

import tensorflow as tf
import numpy as np
import itertools
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, classification_report, confusion_matrix
from tensorflow.keras import layers as L, models as M, applications as A, optimizers as O, callbacks as C

from utils.data_loader_1 import prepare_datasets

IMG_SIZE = (224, 224)
BATCH_SIZE = 64
EPOCHS_PHASE1 = 8    # warm-up
EPOCHS_PHASE2 = 20   # fine-tune (with early stopping)
VAL_SPLIT = 0.2
SEED = 42
NUM_CLASSES = 7

# ---------------------
# Data
# ---------------------
train_ds, val_ds, test_ds, class_names, class_weights = prepare_datasets(
    train_dir="data/train",
    test_dir="data/test",
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    val_split=VAL_SPLIT,
    seed=SEED,
)

print("[info] Classes:", class_names)
print("[info] Class weights:", class_weights)

# ---------------------
# Data augmentation
# (Keep light and realistic; alignment (MTCNN/RetinaFace) would further improve robustness.)
# ---------------------
augmentation = M.Sequential([
    L.RandomFlip("horizontal"),
    L.RandomRotation(0.08),
    L.RandomZoom(0.1),
    L.RandomTranslation(0.05, 0.05),
    L.RandomContrast(0.1),
], name="augmentation")

# ---------------------
# Model
# ---------------------
# NOTE: No extra Rescaling here since VGG16's preprocess_input expects raw 0..255 and does its own scaling.
preprocess = L.Lambda(A.vgg16.preprocess_input, name="vgg16_preprocess")

def build_model(trainable_backbone: bool = False) -> tf.keras.Model:
    inputs = L.Input(shape=(*IMG_SIZE, 3))
    x = augmentation(inputs)
    x = preprocess(x)

    base = A.VGG16(include_top=False, weights="imagenet", input_tensor=x)
    base.trainable = trainable_backbone

    # Global pooling + compact head. BN/Dropout help stability on small/noisy FER datasets
    x = L.GlobalAveragePooling2D(name="gap")(base.output)
    x = L.Dense(256, activation="relu")(x)
    x = L.BatchNormalization()(x)
    x = L.Dropout(0.4)(x)
    x = L.Dense(128, activation="relu")(x)
    x = L.BatchNormalization()(x)
    x = L.Dropout(0.3)(x)
    outputs = L.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)

    return M.Model(inputs, outputs, name="VGG16_FER")

model = build_model(trainable_backbone=False)  # Phase 1: frozen backbone

# ---------------------
# Compile (Phase 1)
# ---------------------
# Use categorical_crossentropy (standard for 7-way softmax).
# Adam with moderate LR; we'll lower LR for fine-tuning.
model.compile(
    optimizer=O.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Callbacks
ckpt_path = "checkpoints/vgg16_fer_phase1.keras"
os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

callbacks_phase1 = [
    C.ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1),
    C.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1, min_lr=1e-6),
    C.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
]

print("[info] Phase 1: training frozen backbone...")
history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_PHASE1,
    class_weight=class_weights,
    callbacks=callbacks_phase1,
    verbose=1,
)

# ---------------------
# Phase 2: Fine-tune top blocks
# ---------------------
# Unfreeze last several convolutional blocks of VGG16:
def unfreeze_top_layers(model: tf.keras.Model, num_blocks_to_unfreeze: int = 2):
    """
    Unfreezes last `num_blocks_to_unfreeze` VGG16 conv blocks.
    VGG16 conv blocks are named: block1_*, block2_*, ..., block5_*
    """
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):  # the base model
            for l in layer.layers:
                l.trainable = False  # freeze all first
            # then unfreeze top blocks
            for block_idx in range(5, 5 - num_blocks_to_unfreeze, -1):
                for l in layer.layers:
                    if l.name.startswith(f"block{block_idx}_"):
                        l.trainable = True

unfreeze_top_layers(model, num_blocks_to_unfreeze=2)

# Re-compile with lower LR for fine-tuning
model.compile(
    optimizer=O.Adam(learning_rate=5e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

ckpt_path2 = "checkpoints/vgg16_fer_phase2.keras"
callbacks_phase2 = [
    C.ModelCheckpoint(ckpt_path2, monitor="val_accuracy", save_best_only=True, verbose=1),
    C.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1, min_lr=1e-7),
    C.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
]

print("[info] Phase 2: fine-tuning top VGG16 blocks...")
history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_PHASE2,
    class_weight=class_weights,
    callbacks=callbacks_phase2,
    verbose=1,
)

# ---------------------
# Evaluation: macro-F1 and confusion matrix on TEST
# ---------------------
def evaluate_and_report(model: tf.keras.Model, ds, ds_name: str):
    y_true = []
    y_pred = []
    for x, y in ds:
        p = model.predict(x, verbose=0)
        y_true.extend(np.argmax(y.numpy(), axis=1))
        y_pred.extend(np.argmax(p, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = np.mean(y_true == y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"\n[{ds_name}] accuracy={acc:.4f}  macro-F1={macro_f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    # Confusion matrix plot
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig = plot_confusion_matrix(cm, class_names, title=f"Confusion Matrix - {ds_name}")
    fig.savefig(f"confusion_matrix_{ds_name.lower()}.png", dpi=150, bbox_inches="tight")
    print(f"[info] Saved confusion matrix: confusion_matrix_{ds_name.lower()}.png")

def plot_confusion_matrix(cm, classes, title="Confusion matrix"):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = "d"
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig

print("\n[info] Evaluating on TEST set...")
evaluate_and_report(model, test_ds, "TEST")

# Save final model
final_path = "models/vgg16_fer_final.keras"
os.makedirs(os.path.dirname(final_path), exist_ok=True)
model.save(final_path)
print(f"[info] Saved model to: {final_path}")
