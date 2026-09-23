from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import sys
import cv2

# ============================================================
# PROJECT SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

app = FastAPI()

# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# DIRECTORIES
# ============================================================

UPLOAD_DIR = PROJECT_ROOT / "uploads"
FRAME_DIR = PROJECT_ROOT / "detection_frames"

UPLOAD_DIR.mkdir(exist_ok=True)
FRAME_DIR.mkdir(exist_ok=True)

# Serve videos and evidence frames
app.mount(
    "/uploads",
    StaticFiles(directory=str(UPLOAD_DIR)),
    name="uploads"
)

app.mount(
    "/detection_frames",
    StaticFiles(directory=str(FRAME_DIR)),
    name="detection_frames"
)


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Accident Detection Backend is running"
    }


# ============================================================
# VIDEO UPLOAD + AI ANALYSIS
# ============================================================

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):

    # --------------------------------------------------------
    # Save uploaded video
    # --------------------------------------------------------

    file_path = UPLOAD_DIR / file.filename

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # --------------------------------------------------------
    # Run existing AI inference
    # --------------------------------------------------------

    from scripts.web_inference import predict_video

    result = predict_video(str(file_path))

    prediction = result["prediction"]
    accident_probability = result["accident_probability"]
    confidence = result["confidence"]
    timestamp_seconds = result["timestamp_seconds"]
    inference_time_seconds = result["inference_time_seconds"]
    sequence_count = result["sequence_count"]
    sms_sent = result.get("sms_sent", False)

    # --------------------------------------------------------
    # APPROXIMATE DETECTION WINDOW
    #
    # 32 frames / 10 FPS = 3.2 seconds
    # --------------------------------------------------------

    SEQUENCE_DURATION = 32 / 10

    approximate_start_seconds = float(timestamp_seconds)

    approximate_end_seconds = (
        approximate_start_seconds + SEQUENCE_DURATION
    )

    # --------------------------------------------------------
    # Extract representative frame from middle
    # of approximate detection window
    # --------------------------------------------------------

    evidence_frame_url = None
    representative_timestamp = None

    if prediction == "ACCIDENT":

        representative_timestamp = (
            approximate_start_seconds
            + (SEQUENCE_DURATION / 2)
        )

        video = cv2.VideoCapture(str(file_path))

        if video.isOpened():

            video.set(
                cv2.CAP_PROP_POS_MSEC,
                representative_timestamp * 1000
            )

            success, frame = video.read()

            if success:

                frame_name = (
                    f"{file_path.stem}_detection.jpg"
                )

                frame_path = FRAME_DIR / frame_name

                cv2.imwrite(
                    str(frame_path),
                    frame
                )

                evidence_frame_url = (
                    f"/detection_frames/{frame_name}"
                )

            video.release()

    # --------------------------------------------------------
    # RETURN RESULT
    # --------------------------------------------------------

    return {
        "prediction": prediction,

        "accident_probability": accident_probability,

        "confidence": confidence,

        "timestamp_seconds": timestamp_seconds,

        "approximate_start_seconds":
            round(approximate_start_seconds, 2),

        "approximate_end_seconds":
            round(approximate_end_seconds, 2),

        "representative_timestamp_seconds":
            round(representative_timestamp, 2)
            if representative_timestamp is not None
            else None,

        "inference_time_seconds":
            inference_time_seconds,

        "sequence_count":
            sequence_count,
            
            "sms_sent":
            sms_sent,

        "evidence_frame_url":
            evidence_frame_url
    }