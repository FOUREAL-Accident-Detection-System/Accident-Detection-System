import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

TARGET_FPS = 10
IMAGE_SIZE = (160, 160)

SEQUENCE_LENGTH = 16

BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 0.0001

RANDOM_SEED = 42

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# PATHS
# ============================================================

# Use leakage-controlled clean preprocessing
PREPROCESSED_DIR = PROJECT_ROOT / "preprocessed_clean"

RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_DIR = PROJECT_ROOT / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

resolution_name = f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}"

METADATA_FILE = (
    RESULTS_DIR /
    f"preprocessing_clean_{TARGET_FPS}fps_{resolution_name}.csv"
)

# Save clean baseline separately
MODEL_PATH = (
    MODEL_DIR /
    f"accident_detection_clean_{TARGET_FPS}fps_{resolution_name}.keras"
)

BEST_MODEL_PATH = (
    MODEL_DIR /
    f"accident_detection_clean_{TARGET_FPS}fps_{resolution_name}_best.keras"
)

METRICS_FILE = (
    RESULTS_DIR /
    f"metrics_clean_{TARGET_FPS}fps_{resolution_name}.json"
)

CONFUSION_FILE = (
    RESULTS_DIR /
    f"confusion_matrix_clean_{TARGET_FPS}fps_{resolution_name}.csv"
)

HISTORY_FILE = (
    RESULTS_DIR /
    f"training_history_clean_{TARGET_FPS}fps_{resolution_name}.csv"
)


# ============================================================
# DATA GENERATOR
# ============================================================

class SequenceGenerator(tf.keras.utils.Sequence):

    def __init__(self, dataframe, batch_size=4, shuffle=False):

        self.dataframe = dataframe.reset_index(drop=True)
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.indices = np.arange(len(self.dataframe))

        self.on_epoch_end()

    def __len__(self):

        return int(
            np.ceil(
                len(self.dataframe) / self.batch_size
            )
        )

    def __getitem__(self, index):

        batch_indices = self.indices[
            index * self.batch_size:
            (index + 1) * self.batch_size
        ]

        batch = self.dataframe.iloc[batch_indices]

        X = np.zeros(
            (
                len(batch),
                SEQUENCE_LENGTH,
                IMAGE_SIZE[1],
                IMAGE_SIZE[0],
                3
            ),
            dtype=np.float32
        )

        y = np.zeros(
            len(batch),
            dtype=np.float32
        )

        for i, (_, row) in enumerate(batch.iterrows()):

            sequence_path = PROJECT_ROOT / row["sequence_file"]

            data = np.load(sequence_path)

            frames = data["frames"]

            X[i] = frames.astype(np.float32) / 255.0

            y[i] = row["label"]

        return X, y

    def on_epoch_end(self):

        if self.shuffle:
            np.random.shuffle(self.indices)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("CLEAN BASELINE ACCIDENT DETECTION MODEL TRAINING")
print("=" * 70)

print(f"FPS              : {TARGET_FPS}")
print(f"Image size       : {IMAGE_SIZE}")
print(f"Sequence length  : {SEQUENCE_LENGTH}")
print(f"Batch size       : {BATCH_SIZE}")
print(f"Epochs           : {EPOCHS}")
print(f"Learning rate    : {LEARNING_RATE}")

print("\nUsing leakage-controlled dataset split:")
print("dataset_split_clean")

print("=" * 70)


# ============================================================
# CHECK METADATA
# ============================================================

if not METADATA_FILE.exists():

    print("\nERROR:")
    print("Clean preprocessed metadata file was not found.")

    print("\nExpected file:")
    print(METADATA_FILE)

    print("\nRun clean preprocessing first.")

    raise FileNotFoundError(METADATA_FILE)


# ============================================================
# LOAD METADATA
# ============================================================

metadata = pd.read_csv(METADATA_FILE)

train_df = metadata[
    metadata["split"] == "train"
].reset_index(drop=True)

val_df = metadata[
    metadata["split"] == "validation"
].reset_index(drop=True)

test_df = metadata[
    metadata["split"] == "test"
].reset_index(drop=True)


print("\nDataset:")

print(
    f"Training sequences   : {len(train_df)}"
)

print(
    f"Validation sequences : {len(val_df)}"
)

print(
    f"Test sequences       : {len(test_df)}"
)


# ============================================================
# GENERATORS
# ============================================================

train_generator = SequenceGenerator(
    train_df,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_generator = SequenceGenerator(
    val_df,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_generator = SequenceGenerator(
    test_df,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# BUILD MODEL
# ============================================================

print("\nBuilding MobileNetV2 + LSTM model...")

cnn = tf.keras.applications.MobileNetV2(
    input_shape=(
        IMAGE_SIZE[1],
        IMAGE_SIZE[0],
        3
    ),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained MobileNetV2
cnn.trainable = False


inputs = tf.keras.Input(
    shape=(
        SEQUENCE_LENGTH,
        IMAGE_SIZE[1],
        IMAGE_SIZE[0],
        3
    )
)

x = tf.keras.layers.TimeDistributed(
    cnn
)(inputs)

x = tf.keras.layers.TimeDistributed(
    tf.keras.layers.GlobalAveragePooling2D()
)(x)

x = tf.keras.layers.LSTM(64)(x)

x = tf.keras.layers.Dense(
    32,
    activation="relu"
)(x)

x = tf.keras.layers.Dropout(0.3)(x)

outputs = tf.keras.layers.Dense(
    1,
    activation="sigmoid"
)(x)

model = tf.keras.Model(
    inputs,
    outputs
)


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

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    BEST_MODEL_PATH,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)


# ============================================================
# TRAIN
# ============================================================

print("\nStarting clean baseline training...")

training_start = time.time()

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[
        checkpoint,
        early_stopping
    ],
    verbose=1
)

training_time = time.time() - training_start


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(MODEL_PATH)

print("\nFinal clean model saved:")
print(MODEL_PATH)

print("\nBest clean model saved:")
print(BEST_MODEL_PATH)


# ============================================================
# TEST EVALUATION
# ============================================================

print("\nEvaluating on clean test dataset...")

inference_start = time.time()

probabilities = model.predict(
    test_generator,
    verbose=1
).flatten()

inference_time = time.time() - inference_start

predictions = (
    probabilities >= 0.5
).astype(int)

true_labels = (
    test_df["label"]
    .astype(int)
    .values
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    true_labels,
    predictions
)

precision = precision_score(
    true_labels,
    predictions,
    zero_division=0
)

recall = recall_score(
    true_labels,
    predictions,
    zero_division=0
)

f1 = f1_score(
    true_labels,
    predictions,
    zero_division=0
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    true_labels,
    predictions,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {

    "experiment": "clean_baseline",

    "dataset": "dataset_split_clean",

    "fps": TARGET_FPS,

    "image_width": IMAGE_SIZE[0],

    "image_height": IMAGE_SIZE[1],

    "sequence_length": SEQUENCE_LENGTH,

    "batch_size": BATCH_SIZE,

    "epochs": EPOCHS,

    "learning_rate": LEARNING_RATE,

    "train_sequences": len(train_df),

    "validation_sequences": len(val_df),

    "test_sequences": len(test_df),

    "accuracy": float(accuracy),

    "precision": float(precision),

    "recall": float(recall),

    "f1_score": float(f1),

    "true_negative": int(tn),

    "false_positive": int(fp),

    "false_negative": int(fn),

    "true_positive": int(tp),

    "training_time_seconds": float(training_time),

    "test_inference_time_seconds": float(inference_time)
}


with open(METRICS_FILE, "w") as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

cm_df = pd.DataFrame(
    cm,

    index=[
        "Actual Normal",
        "Actual Accident"
    ],

    columns=[
        "Predicted Normal",
        "Predicted Accident"
    ]
)

cm_df.to_csv(
    CONFUSION_FILE
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(
    history.history
)

history_df.to_csv(
    HISTORY_FILE,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("CLEAN BASELINE TRAINING AND EVALUATION COMPLETE")
print("=" * 70)

print(f"FPS              : {TARGET_FPS}")
print(f"Resolution       : {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}")

print("\nPerformance:")

print(
    f"Accuracy         : {accuracy:.4f}"
)

print(
    f"Precision        : {precision:.4f}"
)

print(
    f"Recall           : {recall:.4f}"
)

print(
    f"F1 Score         : {f1:.4f}"
)

print("\nConfusion Matrix:")

print(
    f"TN               : {tn}"
)

print(
    f"FP               : {fp}"
)

print(
    f"FN               : {fn}"
)

print(
    f"TP               : {tp}"
)

print("\nTime:")

print(
    f"Training time    : {training_time:.2f} seconds"
)

print(
    f"Test inference   : {inference_time:.2f} seconds"
)

print("\nSaved files:")

print(MODEL_PATH)
print(BEST_MODEL_PATH)
print(METRICS_FILE)
print(CONFUSION_FILE)
print(HISTORY_FILE)

print("=" * 70)