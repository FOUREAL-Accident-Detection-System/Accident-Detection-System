from pathlib import Path
import subprocess
import sys
import os
import json
import shutil
from datetime import datetime


# ============================================================
# MASTER PIPELINE PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

DATASET_DIR = PROJECT_ROOT / "dataset_split_clean"

CONFIG_FILE = PROJECT_ROOT / "configs" / "config.json"

PREPROCESS_SCRIPT = SCRIPTS_DIR / "preprocess_dataset.py"
TRAIN_SCRIPT = SCRIPTS_DIR / "train_model.py"
INFERENCE_SCRIPT = SCRIPTS_DIR / "batch_inference.py"
SCORECARD_SCRIPT = SCRIPTS_DIR / "create_scorecard.py"

INPUT_FILE = PROJECT_ROOT / "input_videos.txt"

AUTOMATION_RESULTS = PROJECT_ROOT / "automation_results"


# ============================================================
# LOAD CONFIGURATION
# ============================================================

def load_config():

    if not CONFIG_FILE.exists():

        print("ERROR: Configuration file not found:")
        print(CONFIG_FILE)

        sys.exit(1)


    try:

        with open(
            CONFIG_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            config = json.load(file)


    except json.JSONDecodeError as error:

        print(
            "ERROR: config.json contains invalid JSON."
        )

        print(error)

        sys.exit(1)


    required_keys = [

        "model_version",
        "model_path",
        "fps",
        "image_size",
        "sequence_length",
        "stride",
        "lstm_units",
        "dense_units",
        "dropout",
        "learning_rate",
        "batch_size",
        "threshold",
        "aggregation"

    ]


    missing_keys = [

        key
        for key in required_keys
        if key not in config

    ]


    if missing_keys:

        print(
            "ERROR: Missing configuration values:"
        )

        for key in missing_keys:

            print(
                f"  - {key}"
            )

        sys.exit(1)


    return config


# ============================================================
# CREATE UNIQUE RUN FOLDER
# ============================================================

def create_run_folder():

    AUTOMATION_RESULTS.mkdir(
        parents=True,
        exist_ok=True
    )


    run_numbers = []


    for folder in AUTOMATION_RESULTS.iterdir():

        if (
            folder.is_dir()
            and folder.name.startswith("run_")
        ):

            try:

                number = int(
                    folder.name.replace(
                        "run_",
                        ""
                    )
                )

                run_numbers.append(
                    number
                )


            except ValueError:

                pass


    if run_numbers:

        next_number = (
            max(run_numbers) + 1
        )

    else:

        next_number = 1


    run_folder = (

        AUTOMATION_RESULTS
        / f"run_{next_number:03d}"

    )


    logs_folder = (
        run_folder / "logs"
    )


    logs_folder.mkdir(
        parents=True,
        exist_ok=False
    )


    return run_folder


# ============================================================
# SAVE RUN CONFIGURATION
# ============================================================

def save_run_config(
    run_folder,
    config
):

    run_config_file = (
        run_folder / "config.json"
    )


    with open(
        run_config_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            config,
            file,
            indent=4
        )


    return run_config_file


# ============================================================
# COPY FILE SAFELY
# ============================================================

def copy_if_exists(
    source,
    destination
):

    source = Path(source)
    destination = Path(destination)


    if source.exists():

        destination.parent.mkdir(
            parents=True,
            exist_ok=True
        )


        shutil.copy2(
            source,
            destination
        )


        print(
            f"Saved run copy: {destination}"
        )

        return True


    return False


# ============================================================
# PRESERVE TRAINING OUTPUTS
# ============================================================

def preserve_training_outputs(
    run_folder,
    config
):

    print()
    print(
        "Preserving training outputs..."
    )


    models_folder = (
        run_folder / "models"
    )

    training_results_folder = (
        run_folder / "training_results"
    )


    models_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    training_results_folder.mkdir(
        parents=True,
        exist_ok=True
    )


    resolution_name = (

        f"{config['image_size'][0]}"
        f"x"
        f"{config['image_size'][1]}"

    )


    model_base = (
        PROJECT_ROOT
        / "models"
        / (
            f"accident_detection_clean_"
            f"{config['fps']}fps_"
            f"{resolution_name}"
        )
    )


    results_base = (
        PROJECT_ROOT
        / "results"
    )


    copied_files = 0


    # --------------------------------------------------------
    # MODEL FILES
    # --------------------------------------------------------

    model_files = [

        model_base.with_suffix(
            ".keras"
        ),

        Path(
            str(model_base)
            + "_best.keras"
        )

    ]


    for source in model_files:

        destination = (
            models_folder
            / source.name
        )


        if copy_if_exists(
            source,
            destination
        ):

            copied_files += 1


    # --------------------------------------------------------
    # TRAINING RESULT FILES
    # --------------------------------------------------------

    result_files = [

        results_base
        / (
            f"metrics_clean_"
            f"{config['fps']}fps_"
            f"{resolution_name}.json"
        ),

        results_base
        / (
            f"confusion_matrix_clean_"
            f"{config['fps']}fps_"
            f"{resolution_name}.csv"
        ),

        results_base
        / (
            f"training_history_clean_"
            f"{config['fps']}fps_"
            f"{resolution_name}.csv"
        )

    ]


    for source in result_files:

        destination = (
            training_results_folder
            / source.name
        )


        if copy_if_exists(
            source,
            destination
        ):

            copied_files += 1


    print(
        f"Training files preserved: "
        f"{copied_files}"
    )


# ============================================================
# HELPER FUNCTION
# ============================================================

def print_step(
    number,
    title
):

    print()

    print(
        "=" * 70
    )

    print(
        f"STEP {number}: {title}"
    )

    print(
        "=" * 70
    )


# ============================================================
# RUN SCRIPT
# ============================================================

def run_script(
    script_path,
    extra_env=None
):

    print(
        f"\nRunning: "
        f"{script_path.name}"
    )

    print(
        "-" * 70
    )


    environment = os.environ.copy()


    if extra_env:

        environment.update(
            extra_env
        )


    result = subprocess.run(

        [
            sys.executable,
            str(script_path)
        ],

        cwd=str(
            PROJECT_ROOT
        ),

        env=environment

    )


    if result.returncode != 0:

        print()

        print(
            f"ERROR: "
            f"{script_path.name} failed."
        )

        print(
            "Pipeline stopped "
            "to protect later steps."
        )

        sys.exit(
            result.returncode
        )


    print()

    print(
        f"Completed: "
        f"{script_path.name}"
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    start_time = datetime.now()


    # ========================================================
    # LOAD CONFIG
    # ========================================================

    config = load_config()


    image_width = (
        config["image_size"][0]
    )

    image_height = (
        config["image_size"][1]
    )


    model_path = (

        PROJECT_ROOT
        / config["model_path"]

    )


    # ========================================================
    # CREATE UNIQUE RUN
    # ========================================================

    run_folder = (
        create_run_folder()
    )


    logs_folder = (
        run_folder / "logs"
    )


    output_file = (
        run_folder
        / "predictions.csv"
    )


    scorecard_file = (
        run_folder
        / "scorecard.xlsx"
    )


    run_config_file = (
        save_run_config(
            run_folder,
            config
        )
    )


    # ========================================================
    # START DISPLAY
    # ========================================================

    print(
        "=" * 70
    )

    print(
        "ACCIDENT DETECTION - "
        "MASTER AUTOMATION PIPELINE"
    )

    print(
        "=" * 70
    )


    print(
        f"Project root : "
        f"{PROJECT_ROOT}"
    )


    print(
        f"Dataset      : "
        f"{DATASET_DIR}"
    )


    print(
        f"Config       : "
        f"{CONFIG_FILE}"
    )


    print(
        f"Run folder   : "
        f"{run_folder}"
    )


    print(
        f"Model        : "
        f"{model_path}"
    )


    print(
        f"Model version: "
        f"{config['model_version']}"
    )


    print(
        f"Started      : "
        f"{start_time.strftime('%Y-%m-%d %H:%M:%S')}"
    )


    # ========================================================
    # SAFETY CHECKS
    # ========================================================

    print_step(
        "0",
        "Safety Checks"
    )


    if not DATASET_DIR.exists():

        print(
            "ERROR: Clean dataset not found:"
        )

        print(
            DATASET_DIR
        )

        sys.exit(1)


    required_folders = [

        DATASET_DIR / "train",
        DATASET_DIR / "validation",
        DATASET_DIR / "test"

    ]


    for folder in required_folders:

        if not folder.exists():

            print(
                "ERROR: Required folder missing:"
            )

            print(folder)

            sys.exit(1)


    for script in [

        PREPROCESS_SCRIPT,
        TRAIN_SCRIPT,
        INFERENCE_SCRIPT,
        SCORECARD_SCRIPT

    ]:

        if not script.exists():

            print(
                "ERROR: Required script missing:"
            )

            print(script)

            sys.exit(1)


    if not INPUT_FILE.exists():

        print(
            "WARNING: input_videos.txt "
            "not found:"
        )

        print(
            INPUT_FILE
        )

        print(
            "Training/preprocessing can still run, "
            "but inference and scorecard will be skipped."
        )


    print(
        "Safety checks passed."
    )


    print(
        "Verified clean dataset will be used."
    )


    print(
        "100-video unseen evaluation remains separate."
    )


    # ========================================================
    # STEP 1 - CONFIGURATION
    # ========================================================

    print_step(
        "1",
        "Loaded Model Configuration"
    )


    print(
        f"Model version  : "
        f"{config['model_version']}"
    )


    print(
        f"FPS            : "
        f"{config['fps']}"
    )


    print(
        f"Resolution     : "
        f"{image_width}x{image_height}"
    )


    print(
        f"Sequence length: "
        f"{config['sequence_length']}"
    )


    print(
        f"Stride         : "
        f"{config['stride']}"
    )


    print(
        f"LSTM units     : "
        f"{config['lstm_units']}"
    )


    print(
        f"Dense units    : "
        f"{config['dense_units']}"
    )


    print(
        f"Dropout        : "
        f"{config['dropout']}"
    )


    print(
        f"Learning rate  : "
        f"{config['learning_rate']}"
    )


    print(
        f"Batch size     : "
        f"{config['batch_size']}"
    )


    print(
        f"Threshold      : "
        f"{config['threshold']}"
    )


    print(
        f"Aggregation    : "
        f"{config['aggregation']}"
    )


    print(
        f"Run config saved: "
        f"{run_config_file}"
    )


    # ========================================================
    # STEP 2 - PREPROCESSING
    # ========================================================

    print_step(
        "2",
        "Automated Preprocessing"
    )


    run_script(
        PREPROCESS_SCRIPT
    )


    # ========================================================
    # STEP 3 - TRAINING
    # ========================================================

    print_step(
        "3",
        "Automated Model Training"
    )


    run_script(
        TRAIN_SCRIPT
    )


    # ========================================================
    # PRESERVE TRAINING OUTPUTS
    # ========================================================

    preserve_training_outputs(
        run_folder,
        config
    )


    # ========================================================
    # STEP 4 - BATCH INFERENCE
    # ========================================================

    if INPUT_FILE.exists():

        print_step(
            "4",
            "Automated Batch Inference"
        )


        inference_environment = {

            "ADP_MODEL_PATH":
                str(model_path),

            "ADP_INPUT_FILE":
                str(INPUT_FILE),

            "ADP_OUTPUT_FILE":
                str(output_file),

            "ADP_LOG_DIRECTORY":
                str(logs_folder),

            "ADP_TARGET_FPS":
                str(config["fps"]),

            "ADP_IMAGE_WIDTH":
                str(image_width),

            "ADP_IMAGE_HEIGHT":
                str(image_height),

            "ADP_SEQUENCE_LENGTH":
                str(
                    config["sequence_length"]
                ),

            "ADP_STRIDE":
                str(
                    config["stride"]
                ),

            "ADP_THRESHOLD":
                str(
                    config["threshold"]
                ),

            "ADP_BATCH_SIZE":
                str(
                    config["batch_size"]
                )

        }


        run_script(

            INFERENCE_SCRIPT,

            extra_env=inference_environment

        )


        # ====================================================
        # STEP 5 - SCORECARD
        # ====================================================

        print_step(
            "5",
            "Automatic Excel Scorecard"
        )


        scorecard_environment = {

            "ADP_PREDICTIONS_FILE":
                str(output_file),

            "ADP_CONFIG_FILE":
                str(run_config_file),

            "ADP_SCORECARD_FILE":
                str(scorecard_file)

        }


        run_script(

            SCORECARD_SCRIPT,

            extra_env=scorecard_environment

        )


    else:

        print_step(
            "4",
            "Batch Inference Skipped"
        )


        print(
            "input_videos.txt was not found."
        )


        print(
            "Scorecard was also skipped."
        )


    # ========================================================
    # FINISH
    # ========================================================

    end_time = datetime.now()


    duration = (

        end_time
        - start_time

    )


    print()

    print(
        "=" * 70
    )

    print(
        "PIPELINE COMPLETED"
    )

    print(
        "=" * 70
    )


    print(
        f"Started : "
        f"{start_time.strftime('%Y-%m-%d %H:%M:%S')}"
    )


    print(
        f"Finished: "
        f"{end_time.strftime('%Y-%m-%d %H:%M:%S')}"
    )


    print(
        f"Duration: "
        f"{duration}"
    )


    print()


    print(
        "Run results saved in:"
    )


    print(
        run_folder
    )


    if INPUT_FILE.exists():

        print()

        print(
            "Predictions:"
        )

        print(
            output_file
        )


        print()

        print(
            "Excel scorecard:"
        )

        print(
            scorecard_file
        )


        print()

        print(
            "Individual logs:"
        )

        print(
            logs_folder
        )


    print()

    print(
        "Training outputs were preserved "
        "inside this run."
    )


    print(
        "Previous automation runs are preserved."
    )


    print(
        "Future Seq32/Seq64 models can use "
        "separate configurations."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()