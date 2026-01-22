import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import winsound
import math

# ===============================
# Mediapipe Setup
# ===============================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

# ===============================
# Load Model
# ===============================
interpreter = tf.lite.Interpreter(model_path="model/model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ===============================
# Camera
# ===============================
cap = cv2.VideoCapture(0)

# Finger tips
finger_tips = [4, 8, 12, 16, 20]

sound_played = False

# ===============================
# Helper Functions
# ===============================
def distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def detect_gesture(hand):
    thumb = hand.landmark[4]
    index = hand.landmark[8]
    middle = hand.landmark[12]
    ring = hand.landmark[16]
    pinky = hand.landmark[20]

    # Distance checks
    pinch_dist = distance(thumb, index)

    fingers_open = [
        hand.landmark[8].y < hand.landmark[6].y,
        hand.landmark[12].y < hand.landmark[10].y,
        hand.landmark[16].y < hand.landmark[14].y,
        hand.landmark[20].y < hand.landmark[18].y
    ]

    if pinch_dist < 0.03:
        return "PINCH 🤏"

    elif all(fingers_open):
        return "OPEN PALM ✋"

    elif not any(fingers_open):
        return "FIST ✊"

    elif thumb.y < hand.landmark[3].y:
        return "THUMBS UP 👍"

    return "UNKNOWN"

# ===============================
# Main Loop
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        h, w, _ = frame.shape

        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        # ML model
        img = cv2.resize(frame, (224, 224)) / 255.0
        img = np.expand_dims(img, axis=0).astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        prediction = np.argmax(output)

        # Gesture detection
        gesture = detect_gesture(hand)

        # Touch logic
        if prediction == 1:
            text = "TOUCH DETECTED"
            color = (0, 255, 0)
            if not sound_played:
                winsound.Beep(1200, 150)
                sound_played = True
        else:
            text = "NO TOUCH"
            color = (0, 0, 255)
            sound_played = False

        # Draw finger dots
        for tip in finger_tips:
            x = int(hand.landmark[tip].x * w)
            y = int(hand.landmark[tip].y * h)
            cv2.circle(frame, (x, y), 10, (255, 0, 0), -1)

        # Text overlays
        cv2.putText(frame, text, (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.putText(frame, f"Gesture: {gesture}", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
