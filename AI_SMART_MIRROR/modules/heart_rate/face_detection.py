import cv2
import mediapipe as mp

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

mp_face = mp.solutions.face_detection

print("Face detection started. Press Q to quit.")

with mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
) as face_detection:

    while True:
        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read frame.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:

                bbox = detection.location_data.relative_bounding_box

                h, w, _ = frame.shape

                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                box_w = int(bbox.width * w)
                box_h = int(bbox.height * h)

                x = max(0, x)
                y = max(0, y)

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + box_w, y + box_h),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    "Face detected",
                    (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

        cv2.imshow("AI Smart Mirror - Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()