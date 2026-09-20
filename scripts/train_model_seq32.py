import os
import json
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Input,
    TimeDistributed,
    GlobalAveragePooling2D,
    LSTM,
    Dense,
    Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)


# ============================================================
# CONFIGURATION — SEQ32 EXPERIMENT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PREPROCESSED_DIR = (
    PROJECT_ROOT / "preprocessed_clean_seq32"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "results"
    / "preprocessing_clean_10fps_160x160_seq32.csv"
)

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

TARGET_FPS = 10
IMAGE_SIZE = (160, 160)

SEQUENCE_LENGTH = 32
STRIDE = 32

BATCH_SIZE = 4
EPOCHS = 10

LEARNING_RATE = 0.0001
RANDOM_SEED = 42


# ============================================================
# OUTPUT FILES
# ============================================================

MODEL_NAME = (
    "accident_detection_clean_"
    "10fps_160x160_seq32"
)

MODEL_PATH = (
    MODELS_DIR / f"{MODEL_NAME}.keras"
)

BEST_MODEL_PATH = (
    MODELS_DIR / f"{MODEL_NAME}_best.keras"
)

METRICS_PATH = (
    RESULTS_DIR / f"metrics_{MODEL_NAME}.json"
)

CONFUSION_PATH = (
    RESULTS_DIR / f"confusion_matrix_{MODEL_NAME}.csv"
)

HISTORY_PATH = (
    RESULTS_DIR / f"training_history_{MODEL_NAME}.csv"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================
# SEQUENCE GENERATOR
# ============================================================

class SequenceGenerator(tf.keras.utils.Sequence):

    def __init__(
        self,
        metadata,
        batch_size,
        shuffle=True
    ):

        self.metadata = (
            metadata.reset_index(drop=True)
        )

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.indices = np.arange(
            len(self.metadata)
        )

        self.on_epoch_end()


    def __len__(self):

        return int(
            np.ceil(
                len(self.metadata)
                / self.batch_size
            )
        )


    def __getitem__(self, index):

        batch_indices = (
            self.indices[
                index * self.batch_size:
                (index + 1) * self.batch_size
            ]
        )

        batch_rows = (
            self.metadata.iloc[
                batch_indices
            ]
        )

        X = []
        y = []

        for _, row in batch_rows.iterrows():

            sequence_path = (
                PROJECT_ROOT
                / row["sequence_file"]
            )

            with np.load(sequence_path) as data:
                frames = data["frames"]

            frames = (
                frames.astype(np.float32)
                / 255.0
            )

            X.append(frames)

            y.append(
                int(row["label"])
            )

        return (
            np.array(X, dtype=np.float32),
            np.array(y, dtype=np.float32)
        )


    def on_epoch_end(self):

        if self.shuffle:

            np.random.shuffle(
                self.indices
            )


# ============================================================
# BUILD MODEL
# ============================================================

def build_model():

    print(
        "\nBuilding MobileNetV2 + LSTM model..."
    )

    input_layer = Input(
        shape=(
            SEQUENCE_LENGTH,
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
            3
        )
    )

    cnn = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
            3
        )
    )

    # Freeze CNN
    cnn.trainable = False

    x = TimeDistributed(
        cnn
    )(input_layer)

    x = TimeDistributed(
        GlobalAveragePooling2D()
    )(x)

    x = LSTM(
        64
    )(x)

    x = Dense(
        32,
        activation="relu"
    )(x)

    x = Dropout(
        0.3
    )(x)

    output = Dense(
        1,
        activation="sigmoid"
    )(x)

    model = Model(
        inputs=input_layer,
        outputs=output
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_probability,
    threshold=0.5
):

    y_pred = (
        y_probability >= threshold
    ).astype(int)

    tp = int(
        np.sum(
            (y_true == 1)
            & (y_pred == 1)
        )
    )

    tn = int(
        np.sum(
            (y_true == 0)
            & (y_pred == 0)
        )
    )

    fp = int(
        np.sum(
            (y_true == 0)
            & (y_pred == 1)
        )
    )

    fn = int(
        np.sum(
            (y_true == 1)
            & (y_pred == 0)
        )
    )

    total = (
        tp + tn + fp + fn
    )

    accuracy = (
        (tp + tn) / total
        if total > 0
        else 0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SEQ32 MODEL TRAINING")
    print("=" * 70)

    print(f"FPS              : {TARGET_FPS}")
    print(f"Image size       : {IMAGE_SIZE}")
    print(f"Sequence length  : {SEQUENCE_LENGTH}")
    print(f"Stride           : {STRIDE}")
    print(f"Batch size       : {BATCH_SIZE}")
    print(f"Epochs           : {EPOCHS}")
    print(f"Learning rate    : {LEARNING_RATE}")
    print(f"Preprocessed dir : {PREPROCESSED_DIR}")
    print(f"Metadata         : {METADATA_PATH}")

    print("=" * 70)


    # ========================================================
    # CHECK INPUTS
    # ========================================================

    if not METADATA_PATH.exists():

        raise FileNotFoundError(
            f"Metadata not found:\n"
            f"{METADATA_PATH}"
        )

    if not PREPROCESSED_DIR.exists():

        raise FileNotFoundError(
            f"Preprocessed data not found:\n"
            f"{PREPROCESSED_DIR}"
        )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # LOAD METADATA
    # ========================================================

    metadata = pd.read_csv(
        METADATA_PATH
    )

    print(
        f"\nTotal sequences: "
        f"{len(metadata)}"
    )

    print(
        "\nSequences by split/class:"
    )

    print(
        metadata
        .groupby(
            ["split", "class"]
        )
        .size()
        .to_string()
    )


    # ========================================================
    # SPLIT METADATA
    # ========================================================

    train_metadata = metadata[
        metadata["split"] == "train"
    ].copy()

    validation_metadata = metadata[
        metadata["split"] == "validation"
    ].copy()

    test_metadata = metadata[
        metadata["split"] == "test"
    ].copy()


    print("\nSequence counts:")

    print(
        f"Train      : "
        f"{len(train_metadata)}"
    )

    print(
        f"Validation : "
        f"{len(validation_metadata)}"
    )

    print(
        f"Test       : "
        f"{len(test_metadata)}"
    )


    # ========================================================
    # CREATE GENERATORS
    # ========================================================

    train_generator = SequenceGenerator(
        train_metadata,
        BATCH_SIZE,
        shuffle=True
    )

    validation_generator = SequenceGenerator(
        validation_metadata,
        BATCH_SIZE,
        shuffle=False
    )

    test_generator = SequenceGenerator(
        test_metadata,
        BATCH_SIZE,
        shuffle=False
    )


    # ========================================================
    # BUILD MODEL
    # ========================================================

    model = build_model()

    model.summary()


    # ========================================================
    # CALLBACKS
    # ========================================================

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        BEST_MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )


    # ========================================================
    # TRAIN
    # ========================================================

    print("\nStarting training...")

    start_time = time.time()

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=[
            early_stopping,
            checkpoint
        ],
        verbose=1
    )

    training_time = (
        time.time() - start_time
    )


    # ========================================================
    # SAVE FINAL MODEL
    # ========================================================

    model.save(
        MODEL_PATH
    )

    print(
        f"\nFinal model saved to:\n"
        f"{MODEL_PATH}"
    )

    print(
        f"Best model saved to:\n"
        f"{BEST_MODEL_PATH}"
    )


    # ========================================================
    # SAVE TRAINING HISTORY
    # ========================================================

    history_df = pd.DataFrame(
        history.history
    )

    history_df.insert(
        0,
        "epoch",
        range(
            1,
            len(history_df) + 1
        )
    )

    history_df.to_csv(
        HISTORY_PATH,
        index=False
    )

    print(
        f"Training history saved to:\n"
        f"{HISTORY_PATH}"
    )


    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    print("\nLoading best model...")

    best_model = (
        tf.keras.models.load_model(
            BEST_MODEL_PATH
        )
    )


    # ========================================================
    # TEST PREDICTIONS
    # ========================================================

    print(
        "\nGenerating test predictions..."
    )

    inference_start = time.time()

    y_probability = (
        best_model.predict(
            test_generator,
            verbose=1
        )
        .ravel()
    )

    inference_time = (
        time.time()
        - inference_start
    )

    y_true = (
        test_metadata["label"]
        .to_numpy()
        .astype(int)
    )


    # ========================================================
    # METRICS
    # ========================================================

    metrics = calculate_metrics(
        y_true,
        y_probability,
        threshold=0.5
    )

    metrics["training_time_seconds"] = (
        float(training_time)
    )

    metrics["test_inference_time_seconds"] = (
        float(inference_time)
    )

    metrics["fps"] = TARGET_FPS
    metrics["image_width"] = IMAGE_SIZE[0]
    metrics["image_height"] = IMAGE_SIZE[1]
    metrics["sequence_length"] = SEQUENCE_LENGTH
    metrics["stride"] = STRIDE
    metrics["batch_size"] = BATCH_SIZE
    metrics["epochs"] = len(
        history_df
    )
    metrics["learning_rate"] = (
        LEARNING_RATE
    )
    metrics["threshold"] = 0.5


    # ========================================================
    # SAVE METRICS
    # ========================================================

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )


    # ========================================================
    # SAVE CONFUSION MATRIX
    # ========================================================

    confusion_df = pd.DataFrame(
        [
            {
                "TN": metrics["TN"],
                "FP": metrics["FP"],
                "FN": metrics["FN"],
                "TP": metrics["TP"]
            }
        ]
    )

    confusion_df.to_csv(
        CONFUSION_PATH,
        index=False
    )


    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "SEQ32 TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Accuracy   : "
        f"{metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision  : "
        f"{metrics['precision'] * 100:.2f}%"
    )

    print(
        f"Recall     : "
        f"{metrics['recall'] * 100:.2f}%"
    )

    print(
        f"F1 Score   : "
        f"{metrics['f1_score'] * 100:.2f}%"
    )

    print(
        f"Training time : "
        f"{training_time:.2f} seconds"
    )

    print(
        f"Test inference : "
        f"{inference_time:.2f} seconds"
    )

    print("\nMetrics saved to:")
    print(METRICS_PATH)

    print("\nConfusion matrix saved to:")
    print(CONFUSION_PATH)

    print("=" * 70)


if __name__ == "__main__":
    main()