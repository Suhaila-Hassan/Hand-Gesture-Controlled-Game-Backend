import pytest
from fastapi.testclient import TestClient
from main import app
import cv2
import mediapipe as mp
import numpy as np

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hand Gesture Recognition API is Running"}

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def detect_hand_landmarks(image_path: str):

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Convert BGR to RGB
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_hand_landmarks:
        return None

    hand_landmarks = results.multi_hand_landmarks[0]

    # Extract (x, y) normalized coordinates for 21 landmarks
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.append([lm.x, lm.y])
    
    hands.close()
    return landmarks

def test_predict_with_image():
    # Put your test image path here (make sure it exists relative to test file)
    image_path = "test_hand.jpg"

    landmarks = detect_hand_landmarks(image_path)
    assert landmarks is not None, "No hand landmarks detected in the test image"

    response = client.post("/predict", json={"landmarks": landmarks})
    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    # Label should be string or None (depending on your model and encoder)
    assert isinstance(data["label"], (str, type(None)))

def test_predict_invalid_shape():
    # Send wrong shape data
    landmarks = [[0.0, 0.0]] * 20  # only 20 landmarks instead of 21
    response = client.post("/predict", json={"landmarks": landmarks})
    assert response.status_code == 200
    assert response.json() == {"label": None}
