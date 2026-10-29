
import cv2
import mediapipe as mp
from collections import deque
from statistics import median
import math

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    raise SystemExit

# Recent measurements smooth out small movements
history = deque(maxlen=15)

# Normal posture reference
baseline = None


def get_measurements(landmarks, width, height):
    """Measure visible head and shoulder geometry."""

    def point(name):
        lm = landmarks[mp_pose.PoseLandmark[name].value]
        return lm.x * width, lm.y * height, lm.visibility

    nose = point("NOSE")
    left_shoulder = point("LEFT_SHOULDER")
    right_shoulder = point("RIGHT_SHOULDER")
    left_ear = point("LEFT_EAR")
    right_ear = point("RIGHT_EAR")

    required = [
        nose, left_shoulder, right_shoulder,
        left_ear, right_ear
    ]

    if any(p[2] < 0.5 for p in required):
        return None

    nx, ny = nose[:2]
    lsx, lsy = left_shoulder[:2]
    rsx, rsy = right_shoulder[:2]
    lex, ley = left_ear[:2]
    rex, rey = right_ear[:2]

    shoulder_width = math.hypot(rsx - lsx, rsy - lsy)
    ear_width = math.hypot(rex - lex, rey - ley)

    if shoulder_width < 1 or ear_width < 1:
        return None

    # Position of the nose relative to the shoulder midpoint.
    # Normalizing by shoulder width helps with camera distance.
    shoulder_mid_x = (lsx + rsx) / 2
    shoulder_mid_y = (lsy + rsy) / 2

    nose_offset_x = (nx - shoulder_mid_x) / shoulder_width
    nose_offset_y = (shoulder_mid_y - ny) / shoulder_width

    # Shoulder line slope (front-view symmetry)
    shoulder_slope = abs(rsy - lsy) / shoulder_width

    # Ear line slope (head tilt estimate)
    head_tilt = abs(rey - ley) / ear_width

    return {
        "nose_x": nose_offset_x,
        "nose_y": nose_offset_y,
        "shoulder_slope": shoulder_slope,
        "head_tilt": head_tilt
    }


try:
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        while True:
            ok, frame = cap.read()

            if not ok:
                print("Could not read webcam frame.")
                break

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            message = "Keep face and shoulders visible"
            color = (0, 0, 255)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

                current = get_measurements(
                    results.pose_landmarks.landmark,
                    width,
                    height
                )

                if current is not None:
                    history.append(current)

                    smooth = {
                        key: median(item[key] for item in history)
                        for key in current
                    }

                    if baseline is None:
                        message = "Sit comfortably upright; press C"
                        color = (0, 255, 255)
                    else:
                        # Changes from the user's own calibrated position
                        head_change = abs(
                            smooth["nose_x"] - baseline["nose_x"]
                        )
                        vertical_change = abs(
                            smooth["nose_y"] - baseline["nose_y"]
                        )
                        shoulder_change = abs(
                            smooth["shoulder_slope"]
                            - baseline["shoulder_slope"]
                        )

                        # These are experimental triggers, not validated
                        # measures of forward-head posture or rounded back.
                        possible_change = (
                            head_change > 0.08
                            or vertical_change > 0.10
                            or shoulder_change > 0.06
                        )

                        if possible_change:
                            message = "Posture changed - check your position"
                            color = (0, 165, 255)
                        else:
                            message = "Near your calibrated position"
                            color = (0, 255, 0)

                    cv2.putText(
                        frame,
                        f"Nose X: {smooth['nose_x']:.2f}",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2
                    )

                    cv2.putText(
                        frame,
                        f"Nose Y: {smooth['nose_y']:.2f}",
                        (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2
                    )

                    cv2.putText(
                        frame,
                        f"Shoulder slope: {smooth['shoulder_slope']:.3f}",
                        (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2
                    )

                else:
                    message = "Move back until both ears and shoulders show"
            else:
                message = "No body detected"

            cv2.putText(
                frame, message, (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )
            cv2.putText(
                frame, "C: calibrate upright | Q: quit",
                (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2
            )

            cv2.imshow("AI Smart Mirror - Posture Test", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("c"):
                if len(history) >= 5:
                    baseline = {
                        key: median(item[key] for item in history)
                        for key in history[0]
                    }
                    print("Upright baseline calibrated:", baseline)
                    history.clear()
                else:
                    print("Keep your face and shoulders visible first.")

            elif key == ord("q"):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("\nPosture test closed.")
