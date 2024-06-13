import cv2
import mediapipe as mp

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# OpenCV setup to capture video from the default webcam
cap = cv2.VideoCapture(0)  # Change the index if you have multiple cameras

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open video capture")
    exit()

# Function to detect gesture based on hand landmarks
def detect_gesture(hand_landmarks):
    # Get the coordinates of the fingertips
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    # Example gestures: Thumbs up, Thumbs down, Victory, Fist, Open hand, Pointing, Rock, OK sign, Peace sign
    if thumb_tip.y < index_tip.y < middle_tip.y < ring_tip.y < pinky_tip.y:
        return "Thumbs Up"
    elif thumb_tip.y > index_tip.y > middle_tip.y > ring_tip.y > pinky_tip.y:
        return "Thumbs Down"
    elif index_tip.y > middle_tip.y > ring_tip.y > pinky_tip.y:
        return "Victory"
    elif thumb_tip.y < index_tip.y and middle_tip.y < ring_tip.y and pinky_tip.y < ring_tip.y:
        return "Fist"
    elif thumb_tip.y > index_tip.y and middle_tip.y > ring_tip.y and pinky_tip.y > ring_tip.y:
        return "Open Hand"
    elif thumb_tip.x < index_tip.x < middle_tip.x < ring_tip.x < pinky_tip.x:
        return "Pointing"
    elif thumb_tip.y > index_tip.y > middle_tip.y > ring_tip.y > pinky_tip.y and thumb_tip.x > index_tip.x > middle_tip.x > ring_tip.x > pinky_tip.x:
        return "Rock"
    elif thumb_tip.x < index_tip.x < middle_tip.x < ring_tip.x < pinky_tip.x and thumb_tip.y < index_tip.y < middle_tip.y < ring_tip.y < pinky_tip.y:
        return "OK Sign"
    elif thumb_tip.x > index_tip.x > middle_tip.x > ring_tip.x > pinky_tip.x and thumb_tip.y < index_tip.y < middle_tip.y < ring_tip.y < pinky_tip.y:
        return "Peace Sign"
    return "Unknown Gesture"

# Main loop to read video frames and process them
while cap.isOpened():
    success, frame = cap.read()  # Read a frame from the webcam
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and detect hands
    result = hands.process(frame_rgb)

    # Draw hand landmarks if any hands are detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw the hand landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Detect gesture based on hand landmarks
            gesture = detect_gesture(hand_landmarks)
            # Display the detected gesture on the frame
            cv2.putText(frame, gesture, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame with landmarks and gesture text
    cv2.imshow('Hand Gesture Recognition', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
