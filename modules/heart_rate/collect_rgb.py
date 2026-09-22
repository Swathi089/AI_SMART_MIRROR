import cv2
import mediapipe as mp
import numpy as np
import csv
import time

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

mp_face_mesh = mp.solutions.face_mesh

# Store RGB values
rgb_data = []

print("Starting RGB signal collection...")
print("Look at the camera and remain still.")
print("Collecting data for 30 seconds...")
print("Press Q to stop early.")

start_time = time.time()

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while True:

        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read frame.")
            break

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:

            landmarks = results.multi_face_landmarks[0].landmark

            h, w, _ = frame.shape

            # Find face boundaries
            x_values = [
                int(landmark.x * w)
                for landmark in landmarks
            ]

            y_values = [
                int(landmark.y * h)
                for landmark in landmarks
            ]

            face_x1 = max(0, min(x_values))
            face_x2 = min(w, max(x_values))

            face_y1 = max(0, min(y_values))
            face_y2 = min(h, max(y_values))

            face_width = face_x2 - face_x1
            face_height = face_y2 - face_y1

            # Forehead ROI
            roi_x1 = face_x1 + int(face_width * 0.20)
            roi_x2 = face_x1 + int(face_width * 0.80)

            roi_y1 = face_y1 + int(face_height * 0.05)
            roi_y2 = face_y1 + int(face_height * 0.22)

            # Extract ROI
            roi = frame[
                roi_y1:roi_y2,
                roi_x1:roi_x2
            ]

            if roi.size > 0:

                # OpenCV uses BGR
                mean_bgr = np.mean(
                    roi,
                    axis=(0, 1)
                )

                blue = mean_bgr[0]
                green = mean_bgr[1]
                red = mean_bgr[2]

                elapsed = time.time() - start_time

                rgb_data.append([
                    elapsed,
                    red,
                    green,
                    blue
                ])

                # Display values
                cv2.putText(
                    frame,
                    f"R: {red:.1f}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"G: {green:.1f}",
                    (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"B: {blue:.1f}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2
                )

                # Draw ROI
                cv2.rectangle(
                    frame,
                    (roi_x1, roi_y1),
                    (roi_x2, roi_y2),
                    (255, 0, 0),
                    2
                )

        elapsed = time.time() - start_time

        cv2.putText(
            frame,
            f"Time: {elapsed:.1f}s / 30s",
            (20, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "AI Smart Mirror - RGB Collection",
            frame
        )

        if elapsed >= 30:
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()

# Save collected RGB data
with open(
    "modules/heart_rate/rgb_data.csv",
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "time",
        "red",
        "green",
        "blue"
    ])

    writer.writerows(rgb_data)

print()
print("RGB collection completed.")
print(f"Samples collected: {len(rgb_data)}")
print("Saved to: modules/heart_rate/rgb_data.csv")