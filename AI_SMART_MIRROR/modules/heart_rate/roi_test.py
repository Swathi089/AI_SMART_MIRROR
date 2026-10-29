import cv2
import mediapipe as mp

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

mp_face_mesh = mp.solutions.face_mesh

print("Forehead ROI test started. Press Q to quit.")

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

        # Mirror the camera
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face landmarks
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:

            landmarks = results.multi_face_landmarks[0].landmark

            h, w, _ = frame.shape

            # Find complete face boundaries
            x_values = []
            y_values = []

            for landmark in landmarks:
                x_values.append(int(landmark.x * w))
                y_values.append(int(landmark.y * h))

            face_x1 = max(0, min(x_values))
            face_x2 = min(w, max(x_values))

            face_y1 = max(0, min(y_values))
            face_y2 = min(h, max(y_values))

            face_width = face_x2 - face_x1
            face_height = face_y2 - face_y1

            # GREEN = complete detected face
            cv2.rectangle(
                frame,
                (face_x1, face_y1),
                (face_x2, face_y2),
                (0, 255, 0),
                2
            )

            # BLUE = forehead rPPG region
            roi_x1 = face_x1 + int(face_width * 0.20)
            roi_x2 = face_x1 + int(face_width * 0.80)

            roi_y1 = face_y1 + int(face_height * 0.05)
            roi_y2 = face_y1 + int(face_height * 0.22)

            # Draw forehead ROI
            cv2.rectangle(
                frame,
                (roi_x1, roi_y1),
                (roi_x2, roi_y2),
                (255, 0, 0),
                2
            )

            cv2.putText(
                frame,
                "FOREHEAD rPPG ROI",
                (roi_x1, max(25, roi_y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

        # Show camera
        cv2.imshow("AI Smart Mirror - rPPG ROI", frame)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()