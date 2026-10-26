
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import os

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------
DURATION = 30
CAMERA_INDEX = 0

OUTPUT_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "rgb_data.csv"
)

# --------------------------------------------------
# INITIALIZE CAMERA AND MEDIAPIPE
# --------------------------------------------------
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    print("Check whether another application is using the camera.")
    raise SystemExit

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

mp_face_mesh = mp.solutions.face_mesh

records = []
start_time = time.perf_counter()
last_timestamp = 0.0

print("Starting 30-second RGB recording...")
print("Keep your face still and look toward the camera.")
print("Use steady, comfortable lighting.")
print("Press Q to stop early.")

# --------------------------------------------------
# FACE MESH AND RGB COLLECTION
# --------------------------------------------------
with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as face_mesh:

    while True:
        elapsed = time.perf_counter() - start_time

        if elapsed >= DURATION:
            break

        ret, frame = cap.read()

        if not ret:
            print("ERROR: Could not read webcam frame.")
            break

        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        status = "Face not detected"

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # Face bounding box
            xs = [int(p.x * width) for p in landmarks]
            ys = [int(p.y * height) for p in landmarks]

            face_x1 = max(0, min(xs))
            face_x2 = min(width, max(xs))
            face_y1 = max(0, min(ys))
            face_y2 = min(height, max(ys))

            face_width = face_x2 - face_x1
            face_height = face_y2 - face_y1

            # Forehead region: central upper face, above eyebrows
            roi_x1 = max(0, face_x1 + int(face_width * 0.25))
            roi_x2 = min(width, face_x1 + int(face_width * 0.75))
            roi_y1 = max(0, face_y1 + int(face_height * 0.04))
            roi_y2 = min(height, face_y1 + int(face_height * 0.18))

            if (
                face_width > 0
                and face_height > 0
                and roi_x2 > roi_x1
                and roi_y2 > roi_y1
            ):
                roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

                if roi.size > 0:
                    # Reject very dark or saturated regions
                    brightness = float(np.mean(roi))
                    saturation_fraction = np.mean(
                        np.any(roi >= 250, axis=2)
                    )

                    if brightness > 20 and saturation_fraction < 0.25:
                        # OpenCV frames use BGR channel order
                        blue, green, red = np.mean(
                            roi, axis=(0, 1)
                        )

                        timestamp = time.perf_counter() - start_time

                        # Avoid duplicate/non-increasing timestamps
                        if timestamp > last_timestamp:
                            records.append({
                                "time": timestamp,
                                "red": float(red),
                                "green": float(green),
                                "blue": float(blue)
                            })
                            last_timestamp = timestamp

                        status = "Recording RGB"

                    else:
                        status = "Adjust lighting"

                    cv2.rectangle(
                        frame,
                        (roi_x1, roi_y1),
                        (roi_x2, roi_y2),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        "Forehead ROI",
                        (roi_x1, max(20, roi_y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2
                    )

        remaining = max(0, DURATION - elapsed)

        cv2.putText(
            frame,
            f"{status} | Time left: {remaining:.1f}s",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow("Heart Rate - RGB Collection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Recording stopped by user.")
            break

cap.release()
cv2.destroyAllWindows()

# --------------------------------------------------
# SAVE DATA
# --------------------------------------------------
if len(records) < 100:
    print(f"ERROR: Only {len(records)} valid samples collected.")
    print("Check the webcam, face detection and lighting.")
    raise SystemExit

df = pd.DataFrame(records)
df.to_csv(OUTPUT_FILE, index=False)

print("\nRecording completed.")
print(f"Samples collected: {len(df)}")
print(f"Duration: {df['time'].iloc[-1]:.2f} seconds")
print(f"Saved to: {OUTPUT_FILE}")
print("\nFirst five rows:")
print(df.head().to_string(index=False))
