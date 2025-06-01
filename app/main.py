import numpy as np
import joblib
import warnings
import os
from pydantic import BaseModel
from collections import deque, Counter
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse, Response
from prometheus_client import REGISTRY
from prometheus_client import Counter as PromCounter, Histogram, generate_latest, CONTENT_TYPE_LATEST

warnings.filterwarnings("ignore")

# Load model and encoder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'models', 'XGBoost_best_model.pkl')
encoder_path = os.path.join(BASE_DIR, 'models', 'encoder.pkl')
model = joblib.load(model_path)
encoder = joblib.load(encoder_path)

window_size = 15
gesture_history = deque(maxlen=window_size)

app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Input data model
class LandmarkRequest(BaseModel):
    landmarks: list[list[float]]

# Prometheus metrics
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Time taken for model prediction in seconds"
)

TOTAL_PREDICT_REQUESTS = PromCounter(
    "http_requests_total",
    "Total number of POST requests to /predict endpoint"
)

INVALID_INPUTS = PromCounter(
    "invalid_input_requests_total",
    "Total number of invalid input requests (wrong shape)"
)

EMPTY_LANDMARK_REQUESTS = PromCounter(
    "empty_landmark_requests_total",
    "Total number of empty landmark input requests"
)

PREDICTED_GESTURES = PromCounter(
    "predicted_gestures_total",
    "Total number of predicted gestures",
    ["gesture"]
)

SUCCESSFUL_PREDICTIONS = PromCounter(
    "successful_predictions_total",
    "Total number of successful predictions"
)

MODEL_LOADING_TIME = Histogram(
    "model_loading_time_seconds",
    "Time taken to load models at startup"
)

GESTURE_CONFIDENCE = Histogram(
    "gesture_confidence_score",
    "Confidence score of gesture predictions",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# API endpoints
@app.get("/")
def read_root():
    return {"message": "Hand Gesture Recognition API Running"}

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/metrics")
def metrics():
    """Return Prometheus metrics in the correct format"""
    data = generate_latest(REGISTRY)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

@app.post("/predict")
def predict_landmarks(request: LandmarkRequest):
    TOTAL_PREDICT_REQUESTS.inc()

    if not request.landmarks:
        EMPTY_LANDMARK_REQUESTS.inc()
        return {"label": None, "error": "Empty landmarks"}

    landmarks = np.array(request.landmarks)

    if landmarks.shape != (21, 2):
        INVALID_INPUTS.inc()
        return {"label": None, "error": f"Invalid shape: expected (21, 2), got {landmarks.shape}"}

    try:
        # Normalize landmarks relative to points 0 and 12
        x_scaler = landmarks[12, 0] - landmarks[0, 0]
        y_scaler = landmarks[12, 1] - landmarks[0, 1]
        x_scaler = x_scaler if x_scaler != 0 else 1e-6
        y_scaler = y_scaler if y_scaler != 0 else 1e-6

        landmarks[:, 0] = (landmarks[:, 0] - landmarks[0, 0]) / x_scaler
        landmarks[:, 1] = (landmarks[:, 1] - landmarks[0, 1]) / y_scaler

        features = landmarks.flatten()

        with PREDICTION_LATENCY.time():
            prediction = model.predict([features])[0]
            # Get prediction probabilities for confidence score
            prediction_proba = model.predict_proba([features])[0]
            confidence = np.max(prediction_proba)

        label = encoder.inverse_transform([prediction])[0]

        gesture_history.append(label)
        gesture_counts = Counter(gesture_history)
        gesture = gesture_counts.most_common(1)[0][0]

        label_map = {
            "like": "up",
            "dislike": "down", 
            "stop_inverted": "left",
            "stop": "right"
        }

        direction = label_map.get(gesture, None)

        if direction:
            PREDICTED_GESTURES.labels(gesture=direction).inc()
            SUCCESSFUL_PREDICTIONS.inc()
            GESTURE_CONFIDENCE.observe(confidence)

        return {"label": direction}
        
    except Exception as e:
        # Log the error and increment error counter
        print(f"Prediction error: {str(e)}")
        return {"label": None, "error": f"Prediction failed: {str(e)}"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
