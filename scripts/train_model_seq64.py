import os
import json
import time
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    TimeDistributed,
    GlobalAveragePooling2D,
    LSTM,
    Dense,
    Dropout
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

PREPROCESSED_DIR = os.path.join(
    PROJECT_ROOT,
    "preprocessed_clean_seq64"
)

METADATA_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "preprocessing_clean_10fps_160x160_seq64.csv"
)

MODELS_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_FPS = 10
IMAGE_SIZE = (160, 160)

SEQUENCE_LENGTH = 64
STRIDE = 64

BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 0.0001

RANDOM_SEED = 42


# ============================================================
# OUTPUT FILES
# ============================================================

MODEL_PATH = os.path.join(
    MODELS_DIR,
    "accident_detection_clean_10fps_160x160_seq64.keras"
)

BEST_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "accident_detection_clean_10fps_160x160_seq64_best.keras"
)

METRICS_PATH = os.path.join(
    RESULTS_DIR,
    "metrics_accident_detection_clean_10fps_160x160_seq64.json"
)

CONFUSION_MATRIX_PATH = os.path.join(
    RESULTS_DIR,
    "confusion_matrix_accident_detection_clean_10fps_160x160_seq64.csv"
)

HISTORY_PATH = os.path.join(
    RESULTS_DIR,
    "training_history_accident_detection_clean_10fps_160x160_seq64.csv"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================
# SEQUENCE GENERATOR
# ============================================================

class SequenceGenerator(tf.keras.utils.Sequence):

    def __init__(self, metadata, batch_size=4, shuffle=False):
        super().__init__()

        self.metadata = metadata.reset_index(drop=True)
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.indices = np.arange(len(self.metadata))

        self.on_epoch_end()

    def __len__(self):
        return int(
            np.ceil(
                len(self.metadata) / self.batch_size
            )
        )

    def __getitem__(self, index):

        batch_indices = self.indices[
            index * self.batch_size:
            (index + 1) * self.batch_size
        ]

        batch_rows = self.metadata.iloc[batch_indices]

        X = []
        y = []

        for _, row in batch_rows.iterrows():

            sequence_path = os.path.join(
                PROJECT_ROOT,
                row["sequence_file"]
            )

            with np.load(sequence_path) as data:
                frames = data["frames"]

            frames = frames.astype(
                np.float32
            ) / 255.0

            X.append(frames)
            y.append(float(row["label"]))

        return (
            np.asarray(X, dtype=np.float32),
            np.asarray(y, dtype=np.float32)
        )

    def on_epoch_end(self):

        if self.shuffle:
            np.random.shuffle(self.indices)


# ============================================================
# LOAD METADATA
# ============================================================

print("=" * 70)
print("SEQ64 MODEL TRAINING")
print("=" * 70)

print(f"Target FPS          : {TARGET_FPS}")
print(f"Image size          : {IMAGE_SIZE}")
print(f"Sequence length     : {SEQUENCE_LENGTH}")
print(f"Sequence stride     : {STRIDE}")
print(f"Batch size          : {BATCH_SIZE}")
print(f"Epochs              : {EPOCHS}")
print(f"Learning rate       : {LEARNING_RATE}")


metadata = pd.read_csv(METADATA_PATH)

train_metadata = metadata[
    metadata["split"].str.lower() == "train"
].copy()

validation_metadata = metadata[
    metadata["split"].str.lower() == "validation"
].copy()

test_metadata = metadata[
    metadata["split"].str.lower() == "test"
].copy()


print("\nSequence counts:")
print(f"Train      : {len(train_metadata)}")
print(f"Validation : {len(validation_metadata)}")
print(f"Test       : {len(test_metadata)}")


# ============================================================
# GENERATORS
# ============================================================

train_generator = SequenceGenerator(
    train_metadata,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_generator = SequenceGenerator(
    validation_metadata,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_generator = SequenceGenerator(
    test_metadata,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# BUILD MODEL
# ============================================================

print("\nBuilding MobileNetV2 + LSTM model...")

cnn = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3
    )
)

cnn.trainable = False


model = Sequential([
    TimeDistributed(
        cnn,
        input_shape=(
            SEQUENCE_LENGTH,
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
            3
        )
    ),

    TimeDistributed(
        GlobalAveragePooling2D()
    ),

    LSTM(64),

    Dense(
        32,
        activation="relu"
    ),

    Dropout(0.3),

    Dense(
        1,
        activation="sigmoid"
    )
])


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


model.summary()


# ============================================================
# CALLBACKS
# ============================================================

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


# ============================================================
# TRAIN
# ============================================================

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

training_time = time.time() - start_time


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(MODEL_PATH)

print("\nFinal model saved to:")
print(MODEL_PATH)

print("\nBest model saved to:")
print(BEST_MODEL_PATH)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(history.history)

history_df.insert(
    0,
    "epoch",
    range(1, len(history_df) + 1)
)

history_df.to_csv(
    HISTORY_PATH,
    index=False
)

print("\nTraining history saved to:")
print(HISTORY_PATH)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best model...")

best_model = tf.keras.models.load_model(
    BEST_MODEL_PATH
)


# ============================================================
# TEST PREDICTIONS
# ============================================================

print("\nGenerating test predictions...")

inference_start = time.time()

probabilities = best_model.predict(
    test_generator,
    verbose=1
).ravel()

test_inference_time = (
    time.time() - inference_start
)


# ============================================================
# TEST LABELS
# ============================================================

y_true = test_metadata["label"].astype(int).values

y_pred = (
    probabilities >= 0.50
).astype(int)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

tn, fp, fn, tp = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
).ravel()


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {
    "model": "accident_detection_clean_10fps_160x160_seq64",
    "fps": TARGET_FPS,
    "image_size": list(IMAGE_SIZE),
    "sequence_length": SEQUENCE_LENGTH,
    "stride": STRIDE,
    "batch_size": BATCH_SIZE,
    "epochs_requested": EPOCHS,
    "epochs_completed": len(history.history["loss"]),
    "learning_rate": LEARNING_RATE,
    "threshold": 0.50,
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "TN": int(tn),
    "FP": int(fp),
    "FN": int(fn),
    "TP": int(tp),
    "training_time_seconds": float(training_time),
    "test_inference_time_seconds": float(
        test_inference_time
    )
}

with open(
    METRICS_PATH,
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

confusion_df = pd.DataFrame(
    [
        [tn, fp],
        [fn, tp]
    ],
    index=[
        "Actual Normal",
        "Actual Accident"
    ],
    columns=[
        "Predicted Normal",
        "Predicted Accident"
    ]
)

confusion_df.to_csv(
    CONFUSION_MATRIX_PATH
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("SEQ64 TRAINING COMPLETE")
print("=" * 70)

print(f"Accuracy   : {accuracy * 100:.2f}%")
print(f"Precision  : {precision * 100:.2f}%")
print(f"Recall     : {recall * 100:.2f}%")
print(f"F1 Score   : {f1 * 100:.2f}%")

print(f"TN : {tn}")
print(f"FP : {fp}")
print(f"FN : {fn}")
print(f"TP : {tp}")

print(
    f"Training time : "
    f"{training_time:.2f} seconds"
)

print(
    f"Test inference : "
    f"{test_inference_time:.2f} seconds"
)

print("\nMetrics saved to:")
print(METRICS_PATH)

print("\nConfusion matrix saved to:")
print(CONFUSION_MATRIX_PATH)

print("=" * 70)