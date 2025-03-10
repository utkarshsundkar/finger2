from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import mediapipe as mp

app = FastAPI()

# Initialize MediaPipe Hand Detection
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def count_fingers(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        return 0  # No hands detected

    count = 0
    for hand_landmarks in results.multi_hand_landmarks:
        # Thumb
        if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
            count += 1
        # Fingers
        for tip in [8, 12, 16, 20]:  # Index, Middle, Ring, Pinky
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
                count += 1
    return count

@app.post("/count-fingers/")
async def count_fingers_api(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    fingers = count_fingers(image)
    return {"fingers": fingers}

@app.get("/")
def root():
    return {"message": "Finger counting API is running!"}
