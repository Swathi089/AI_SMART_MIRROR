import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import butter, filtfilt
from collections import deque
import time


# ============================================================
# SETTINGS
# ============================================================

WINDOW_SECONDS = 30
IGNORE_SECONDS = 8

LOW_CUTOFF = 0.7
HIGH_CUTOFF = 3.0

MIN_BPM = 45
MAX_BPM = 140

UPDATE_INTERVAL = 3.0

# How close a new estimate must be to the previous estimate
TRACKING_RANGE = 15.0

# Number of accepted BPM values used for display smoothing
SMOOTHING_COUNT = 5


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open webcam.")
    exit()


# ============================================================
# MEDIAPIPE
# ============================================================

mp_face_mesh = mp.solutions.face_mesh


# ============================================================
# BUFFERS
# ============================================================

time_buffer = deque()
rgb_buffer = deque()

bpm_history = deque(
    maxlen=SMOOTHING_COUNT
)

last_bpm = None
last_update = 0

start_time = time.time()


print()
print("======================================")
print("AI SMART MIRROR - LIVE HEART RATE")
print("======================================")
print("Look at the camera and remain still.")
print("First 8 seconds are ignored.")
print("First result appears after 30 seconds.")
print("Press Q to quit.")
print()


# ============================================================
# CALCULATE HEART RATE
# ============================================================

def calculate_heart_rate(
    time_values,
    rgb_values,
    previous_bpm=None
):

    time_values = np.array(
        time_values
    )

    rgb_values = np.array(
        rgb_values
    )


    # --------------------------------------------------------
    # Minimum data
    # --------------------------------------------------------

    if len(time_values) < 300:
        return None


    # --------------------------------------------------------
    # Ignore first 8 seconds of CURRENT WINDOW
    # --------------------------------------------------------

    relative_time = (
        time_values
        -
        time_values[0]
    )

    mask = (
        relative_time
        >=
        IGNORE_SECONDS
    )

    time_values = time_values[mask]

    rgb_values = rgb_values[mask]


    if len(time_values) < 200:
        return None


    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    differences = np.diff(
        time_values
    )

    if len(differences) == 0:
        return None

    fps = (
        1.0
        /
        np.median(differences)
    )


    if fps <= 0:
        return None


    # --------------------------------------------------------
    # RGB matrix
    # --------------------------------------------------------

    rgb = rgb_values.T


    # ========================================================
    # WINDOWED POS
    # ========================================================

    POS_WINDOW_SECONDS = 1.6

    window_size = int(
        POS_WINDOW_SECONDS
        *
        fps
    )

    if window_size < 10:
        return None

    step = window_size // 2


    pulse_signal = np.zeros(
        len(time_values)
    )

    weight_signal = np.zeros(
        len(time_values)
    )


    for start in range(
        0,
        len(time_values)
        -
        window_size,
        step
    ):

        end = (
            start
            +
            window_size
        )


        window = rgb[
            :,
            start:end
        ]


        # ----------------------------------------------------
        # Temporal normalization
        # ----------------------------------------------------

        mean_rgb = np.mean(
            window,
            axis=1,
            keepdims=True
        )


        if np.any(
            mean_rgb <= 0
        ):
            continue


        normalized = (
            window
            /
            mean_rgb
        )


        # ----------------------------------------------------
        # POS projection
        # ----------------------------------------------------

        s1 = (
            normalized[1]
            -
            normalized[2]
        )


        s2 = (
            -2
            *
            normalized[0]
            +
            normalized[1]
            +
            normalized[2]
        )


        # ----------------------------------------------------
        # Alpha
        # ----------------------------------------------------

        std_s1 = np.std(s1)
        std_s2 = np.std(s2)


        if std_s2 < 1e-8:
            continue


        alpha = (
            std_s1
            /
            std_s2
        )


        # ----------------------------------------------------
        # Pulse
        # ----------------------------------------------------

        pulse = (
            s1
            +
            alpha * s2
        )


        pulse -= np.mean(
            pulse
        )


        # ----------------------------------------------------
        # Overlap-add
        # ----------------------------------------------------

        pulse_signal[
            start:end
        ] += pulse

        weight_signal[
            start:end
        ] += 1


    # ========================================================
    # NORMALIZE OVERLAP
    # ========================================================

    valid = (
        weight_signal > 0
    )


    if np.sum(valid) < 200:
        return None


    pulse_signal[valid] /= (
        weight_signal[valid]
    )


    valid_indices = np.where(
        valid
    )[0]


    first = valid_indices[0]
    last = valid_indices[-1]


    pulse = pulse_signal[
        first:last + 1
    ]


    # ========================================================
    # BAND-PASS FILTER
    # ========================================================

    nyquist = fps / 2


    low = (
        LOW_CUTOFF
        /
        nyquist
    )

    high = (
        HIGH_CUTOFF
        /
        nyquist
    )


    if (
        low <= 0
        or
        high >= 1
    ):
        return None


    b, a = butter(
        3,
        [low, high],
        btype="bandpass"
    )


    if len(pulse) < 100:
        return None


    filtered = filtfilt(
        b,
        a,
        pulse
    )


    filtered -= np.mean(
        filtered
    )


    # ========================================================
    # HANN WINDOW
    # ========================================================

    filtered *= np.hanning(
        len(filtered)
    )


    # ========================================================
    # FFT
    # ========================================================

    n = len(filtered)


    frequencies = np.fft.rfftfreq(
        n,
        d=1 / fps
    )


    magnitudes = np.abs(
        np.fft.rfft(
            filtered
        )
    )


    # ========================================================
    # VALID FREQUENCY RANGE
    # ========================================================

    valid_frequency = (
        (frequencies >= LOW_CUTOFF)
        &
        (frequencies <= HIGH_CUTOFF)
    )


    frequencies = frequencies[
        valid_frequency
    ]

    magnitudes = magnitudes[
        valid_frequency
    ]


    if len(magnitudes) == 0:
        return None


    # ========================================================
    # FIND CANDIDATE PEAKS
    # ========================================================

    candidate_indices = np.argsort(
        magnitudes
    )[::-1]


    candidates = []


    for index in candidate_indices:

        frequency = (
            frequencies[index]
        )

        bpm = (
            frequency * 60
        )

        magnitude = (
            magnitudes[index]
        )


        if (
            bpm < MIN_BPM
            or
            bpm > MAX_BPM
        ):
            continue


        # ----------------------------------------------------
        # Avoid selecting nearby FFT bins as separate peaks
        # ----------------------------------------------------

        too_close = False


        for existing_bpm, _ in candidates:

            if abs(
                bpm
                -
                existing_bpm
            ) < 4:

                too_close = True
                break


        if too_close:
            continue


        candidates.append(
            (
                bpm,
                magnitude
            )
        )


        if len(candidates) >= 5:
            break


    if len(candidates) == 0:
        return None


    # ========================================================
    # SELECT PEAK
    # ========================================================

    selected_bpm = None


    # --------------------------------------------------------
    # If we already have a BPM, prefer a nearby candidate
    # --------------------------------------------------------

    if previous_bpm is not None:

        nearby = []

        for bpm, magnitude in candidates:

            difference = abs(
                bpm
                -
                previous_bpm
            )


            if difference <= TRACKING_RANGE:

                nearby.append(
                    (
                        difference,
                        bpm,
                        magnitude
                    )
                )


        if nearby:

            # Closest candidate to previous BPM
            nearby.sort(
                key=lambda x: x[0]
            )

            selected_bpm = nearby[0][1]


    # --------------------------------------------------------
    # No previous BPM or no nearby candidate
    # --------------------------------------------------------

    if selected_bpm is None:

        # Use strongest candidate
        selected_bpm = candidates[0][0]


    return selected_bpm


# ============================================================
# FACE MESH
# ============================================================

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:


    while True:

        success, frame = (
            camera.read()
        )


        if not success:

            print(
                "ERROR: Could not read frame."
            )

            break


        # ----------------------------------------------------
        # Mirror
        # ----------------------------------------------------

        frame = cv2.flip(
            frame,
            1
        )


        # ----------------------------------------------------
        # Convert to RGB
        # ----------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # ----------------------------------------------------
        # Face landmarks
        # ----------------------------------------------------

        results = face_mesh.process(
            rgb_frame
        )


        if results.multi_face_landmarks:

            landmarks = (
                results
                .multi_face_landmarks[0]
                .landmark
            )


            h, w, _ = (
                frame.shape
            )


            # ------------------------------------------------
            # Face boundaries
            # ------------------------------------------------

            x_values = [
                int(
                    landmark.x * w
                )
                for landmark in landmarks
            ]


            y_values = [
                int(
                    landmark.y * h
                )
                for landmark in landmarks
            ]


            face_x1 = max(
                0,
                min(x_values)
            )


            face_x2 = min(
                w,
                max(x_values)
            )


            face_y1 = max(
                0,
                min(y_values)
            )


            face_y2 = min(
                h,
                max(y_values)
            )


            face_width = (
                face_x2
                -
                face_x1
            )


            face_height = (
                face_y2
                -
                face_y1
            )


            # ------------------------------------------------
            # Forehead ROI
            # ------------------------------------------------

            roi_x1 = (
                face_x1
                +
                int(
                    face_width * 0.20
                )
            )


            roi_x2 = (
                face_x1
                +
                int(
                    face_width * 0.80
                )
            )


            roi_y1 = (
                face_y1
                +
                int(
                    face_height * 0.05
                )
            )


            roi_y2 = (
                face_y1
                +
                int(
                    face_height * 0.22
                )
            )


            # ------------------------------------------------
            # Extract ROI
            # ------------------------------------------------

            roi = frame[
                roi_y1:roi_y2,
                roi_x1:roi_x2
            ]


            if roi.size > 0:

                mean_bgr = np.mean(
                    roi,
                    axis=(0, 1)
                )


                red = mean_bgr[2]
                green = mean_bgr[1]
                blue = mean_bgr[0]


                current_time = (
                    time.time()
                    -
                    start_time
                )


                # ------------------------------------------------
                # Store RGB
                # ------------------------------------------------

                time_buffer.append(
                    current_time
                )


                rgb_buffer.append([
                    red,
                    green,
                    blue
                ])


                # ------------------------------------------------
                # Keep latest 30 seconds
                # ------------------------------------------------

                while (
                    len(time_buffer) > 0
                    and
                    current_time
                    -
                    time_buffer[0]
                    >
                    WINDOW_SECONDS
                ):

                    time_buffer.popleft()

                    rgb_buffer.popleft()


                # ------------------------------------------------
                # Calculate HR
                # ------------------------------------------------

                if (
                    current_time
                    >= WINDOW_SECONDS
                    and
                    current_time
                    -
                    last_update
                    >= UPDATE_INTERVAL
                ):

                    previous_bpm = None


                    if len(
                        bpm_history
                    ) > 0:

                        previous_bpm = (
                            float(
                                np.median(
                                    bpm_history
                                )
                            )
                        )


                    bpm = calculate_heart_rate(
                        list(
                            time_buffer
                        ),
                        list(
                            rgb_buffer
                        ),
                        previous_bpm
                    )


                    if bpm is not None:

                        bpm_history.append(
                            bpm
                        )


                        last_bpm = float(
                            np.median(
                                bpm_history
                            )
                        )


                        print(
                            f"Raw BPM: "
                            f"{bpm:.1f} | "
                            f"Displayed BPM: "
                            f"{last_bpm:.1f}"
                        )


                    else:

                        print(
                            "Signal quality insufficient"
                        )


                    last_update = (
                        current_time
                    )


                # ------------------------------------------------
                # Draw ROI
                # ------------------------------------------------

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
                    (
                        roi_x1,
                        max(
                            25,
                            roi_y1 - 10
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2
                )


        # ====================================================
        # DISPLAY
        # ====================================================

        if last_bpm is None:

            display_text = (
                "Heart Rate: Calculating..."
            )

        else:

            display_text = (
                f"Heart Rate: "
                f"{last_bpm:.1f} BPM"
            )


        cv2.putText(
            frame,
            display_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


        elapsed = (
            time.time()
            -
            start_time
        )


        cv2.putText(
            frame,
            f"Time: {elapsed:.0f}s",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Samples: {len(time_buffer)}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        # ----------------------------------------------------
        # Show
        # ----------------------------------------------------

        cv2.imshow(
            "AI Smart Mirror - Live Heart Rate",
            frame
        )


        # ----------------------------------------------------
        # Quit
        # ----------------------------------------------------

        if (
            cv2.waitKey(1) & 0xFF
            ==
            ord("q")
        ):
            break


# ============================================================
# CLEANUP
# ============================================================

camera.release()
cv2.destroyAllWindows()