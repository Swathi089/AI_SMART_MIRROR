import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import time


# ============================================================
# SETTINGS
# ============================================================

WINDOW_SECONDS = 30
MIN_HR = 45
MAX_HR = 180

LOW_FREQ = 0.75
HIGH_FREQ = 3.0

UPDATE_INTERVAL = 2.0

# Maximum allowed change from previous stable HR
MAX_JUMP_BPM = 15

# Number of recent HR values used for smoothing
SMOOTHING_COUNT = 5


# ============================================================
# BANDPASS FILTER
# ============================================================

def bandpass_filter(signal, fps):

    nyquist = 0.5 * fps

    low = LOW_FREQ / nyquist
    high = HIGH_FREQ / nyquist

    b, a = butter(
        3,
        [low, high],
        btype="band"
    )

    return filtfilt(b, a, signal)


# ============================================================
# POS rPPG
# ============================================================

def calculate_pos_signal(rgb_values):

    rgb = np.asarray(rgb_values, dtype=float)

    # Normalize each RGB channel
    mean_rgb = np.mean(rgb, axis=0)

    normalized = rgb / mean_rgb

    R = normalized[:, 0]
    G = normalized[:, 1]
    B = normalized[:, 2]

    # POS projection
    S1 = G - B
    S2 = G + B - 2 * R

    std_s1 = np.std(S1)
    std_s2 = np.std(S2)

    if std_s2 < 1e-8:
        return None

    alpha = std_s1 / std_s2

    pulse = S1 + alpha * S2

    # Remove slow trend
    pulse = pulse - np.mean(pulse)

    return pulse


# ============================================================
# FIND HR CANDIDATES
# ============================================================

def get_hr_candidates(signal, fps):

    filtered = bandpass_filter(signal, fps)

    # Hann window
    window = np.hanning(len(filtered))

    signal_windowed = filtered * window

    # Zero padding improves frequency resolution
    n_fft = 8192

    fft_result = np.fft.rfft(
        signal_windowed,
        n=n_fft
    )

    frequencies = np.fft.rfftfreq(
        n_fft,
        d=1 / fps
    )

    magnitude = np.abs(fft_result)

    # Keep only heart-rate frequencies
    mask = (
        (frequencies >= LOW_FREQ) &
        (frequencies <= HIGH_FREQ)
    )

    freq = frequencies[mask]
    mag = magnitude[mask]

    if len(freq) == 0:
        return None, None, None

    # Find spectral peaks
    peaks, properties = find_peaks(
        mag,
        distance=20
    )

    if len(peaks) == 0:
        return None, None, None

    candidates = []

    for peak in peaks:

        hr = freq[peak] * 60
        power = mag[peak]

        if MIN_HR <= hr <= MAX_HR:

            candidates.append(
                (hr, power)
            )

    if not candidates:
        return None, None, None

    # Sort by power
    candidates.sort(
        key=lambda x: x[1],
        reverse=True
    )

    strongest_hr = candidates[0][0]
    strongest_power = candidates[0][1]

    # Quality:
    # strongest peak compared with average spectrum
    average_power = np.mean(mag)

    if average_power > 0:
        quality = strongest_power / average_power
    else:
        quality = 0

    return candidates, strongest_hr, quality


# ============================================================
# CHOOSE STABLE HEART RATE
# ============================================================

def choose_stable_hr(
    candidates,
    previous_hr
):

    if candidates is None:
        return None

    # --------------------------------------------------------
    # First measurement:
    # use strongest spectral peak
    # --------------------------------------------------------

    if previous_hr is None:

        return candidates[0][0]


    # --------------------------------------------------------
    # After first measurement:
    # prefer candidates close to previous HR
    # --------------------------------------------------------

    valid_candidates = []

    for hr, power in candidates:

        difference = abs(
            hr - previous_hr
        )

        if difference <= MAX_JUMP_BPM:

            valid_candidates.append(
                (hr, power, difference)
            )


    # --------------------------------------------------------
    # If we have a nearby candidate,
    # choose the strongest nearby candidate.
    # --------------------------------------------------------

    if valid_candidates:

        valid_candidates.sort(
            key=lambda x: x[1],
            reverse=True
        )

        return valid_candidates[0][0]


    # --------------------------------------------------------
    # No candidate near previous HR.
    # Keep previous HR instead of jumping.
    # --------------------------------------------------------

    return previous_hr


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("ERROR: Could not open webcam.")
    exit()


# ============================================================
# MEDIAPIPE FACE MESH
# ============================================================

mp_face_mesh = mp.solutions.face_mesh


# ============================================================
# DATA STORAGE
# ============================================================

rgb_data = []

hr_history = []

previous_hr = None

displayed_hr = None

last_update = 0

start_time = time.time()


print()
print("==========================================")
print("AI SMART MIRROR - STABLE LIVE HEART RATE")
print("==========================================")
print("Look at the camera and remain still.")
print("Keep your forehead visible.")
print("Initial measurement requires about 30 seconds.")
print("Press Q to quit.")
print("==========================================")
print()


# ============================================================
# MAIN LOOP
# ============================================================

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


        # Mirror image
        frame = cv2.flip(
            frame,
            1
        )


        # Current time
        current_time = time.time()

        elapsed = current_time - start_time


        # ----------------------------------------------------
        # RGB conversion
        # ----------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # ----------------------------------------------------
        # Face detection
        # ----------------------------------------------------

        results = face_mesh.process(
            rgb_frame
        )


        if results.multi_face_landmarks:

            landmarks = results.multi_face_landmarks[0].landmark

            h, w, _ = frame.shape


            # ------------------------------------------------
            # Face boundaries
            # ------------------------------------------------

            x_values = []

            y_values = []


            for landmark in landmarks:

                x_values.append(
                    int(landmark.x * w)
                )

                y_values.append(
                    int(landmark.y * h)
                )


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
                face_x2 - face_x1
            )

            face_height = (
                face_y2 - face_y1
            )


            # ------------------------------------------------
            # FOREHEAD ROI
            # ------------------------------------------------

            roi_x1 = (
                face_x1 +
                int(face_width * 0.20)
            )

            roi_x2 = (
                face_x1 +
                int(face_width * 0.80)
            )

            roi_y1 = (
                face_y1 +
                int(face_height * 0.05)
            )

            roi_y2 = (
                face_y1 +
                int(face_height * 0.22)
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


                blue = mean_bgr[0]

                green = mean_bgr[1]

                red = mean_bgr[2]


                # Store RGB
                rgb_data.append([
                    elapsed,
                    red,
                    green,
                    blue
                ])


            # ------------------------------------------------
            # Draw face
            # ------------------------------------------------

            cv2.rectangle(

                frame,

                (face_x1, face_y1),

                (face_x2, face_y2),

                (0, 255, 0),

                2

            )


            # ------------------------------------------------
            # Draw forehead ROI
            # ------------------------------------------------

            cv2.rectangle(

                frame,

                (roi_x1, roi_y1),

                (roi_x2, roi_y2),

                (255, 0, 0),

                2

            )


        # ====================================================
        # HEART RATE CALCULATION
        # ====================================================

        if elapsed >= WINDOW_SECONDS:

            if (
                current_time - last_update
                >= UPDATE_INTERVAL
            ):

                last_update = current_time


                # --------------------------------------------
                # Use last 30 seconds
                # --------------------------------------------

                recent_data = []

                cutoff_time = (
                    elapsed -
                    WINDOW_SECONDS
                )


                for row in rgb_data:

                    if row[0] >= cutoff_time:

                        recent_data.append(row)


                if len(recent_data) > 300:

                    recent_data = np.array(
                        recent_data
                    )


                    time_values = recent_data[:, 0]

                    rgb_values = recent_data[:, 1:4]


                    # ----------------------------------------
                    # Estimate FPS
                    # ----------------------------------------

                    duration = (
                        time_values[-1]
                        -
                        time_values[0]
                    )


                    if duration > 0:

                        fps = (
                            len(time_values)
                            /
                            duration
                        )


                        # --------------------------------
                        # POS
                        # --------------------------------

                        pos_signal = calculate_pos_signal(
                            rgb_values
                        )


                        if pos_signal is not None:

                            candidates, strongest_hr, quality = (
                                get_hr_candidates(
                                    pos_signal,
                                    fps
                                )
                            )


                            if candidates is not None:

                                stable_hr = choose_stable_hr(

                                    candidates,

                                    previous_hr

                                )


                                # --------------------------------
                                # Save HR
                                # --------------------------------

                                if stable_hr is not None:

                                    hr_history.append(
                                        stable_hr
                                    )


                                    # Keep last 5
                                    if len(hr_history) > SMOOTHING_COUNT:

                                        hr_history.pop(0)


                                    # Median is safer
                                    # than simple average
                                    displayed_hr = np.median(
                                        hr_history
                                    )


                                    previous_hr = displayed_hr


                                    # --------------------------------
                                    # Confidence
                                    # --------------------------------

                                    if quality >= 8:

                                        confidence = "HIGH"

                                    elif quality >= 5:

                                        confidence = "MEDIUM"

                                    else:

                                        confidence = "LOW"


                                    print(

                                        f"AI Heart Rate: "
                                        f"{stable_hr:.1f} BPM | "
                                        f"Displayed: "
                                        f"{displayed_hr:.1f} BPM | "
                                        f"Confidence: "
                                        f"{confidence}"

                                    )


        # ====================================================
        # DISPLAY
        # ====================================================

        if displayed_hr is not None:

            cv2.putText(

                frame,

                f"Heart Rate: {displayed_hr:.1f} BPM",

                (20, 40),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (0, 255, 0),

                2

            )

        else:

            cv2.putText(

                frame,

                "Measuring...",

                (20, 40),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (0, 255, 255),

                2

            )


        # Timer

        cv2.putText(

            frame,

            f"Time: {elapsed:.0f}s",

            (20, 75),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            (255, 255, 255),

            2

        )


        cv2.imshow(

            "AI Smart Mirror - Heart Rate",

            frame

        )


        # Q to quit

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break


# ============================================================
# CLEANUP
# ============================================================

camera.release()

cv2.destroyAllWindows()

print()
print("Heart-rate measurement stopped.")