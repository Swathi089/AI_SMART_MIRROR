import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Load RGB data
# --------------------------------------------------

file_path = "modules/heart_rate/rgb_data.csv"

data = pd.read_csv(file_path)

time = data["time"].values
red = data["red"].values
green = data["green"].values
blue = data["blue"].values

print("Total samples:", len(data))


# --------------------------------------------------
# 2. Remove unstable starting portion
# --------------------------------------------------

START_TIME = 8.0

mask = time >= START_TIME

time = time[mask]
red = red[mask]
green = green[mask]
blue = blue[mask]

print("Samples after removing first 8 seconds:", len(time))


# --------------------------------------------------
# 3. Estimate sampling frequency
# --------------------------------------------------

time_difference = np.diff(time)

fps = 1.0 / np.median(time_difference)

print(f"Estimated FPS: {fps:.2f}")


# --------------------------------------------------
# 4. Create RGB matrix
# --------------------------------------------------

rgb = np.vstack([
    red,
    green,
    blue
])


# --------------------------------------------------
# 5. Normalize RGB channels
# --------------------------------------------------

mean_rgb = np.mean(rgb, axis=1, keepdims=True)

rgb_normalized = rgb / mean_rgb


# --------------------------------------------------
# 6. POS algorithm
# --------------------------------------------------

# Projection 1
s1 = rgb_normalized[1] - rgb_normalized[2]

# Projection 2
s2 = (
    rgb_normalized[1]
    + rgb_normalized[2]
    - 2 * rgb_normalized[0]
)


# Calculate alpha
std_s1 = np.std(s1)
std_s2 = np.std(s2)

if std_s2 == 0:
    print("ERROR: Invalid RGB signal.")
    exit()

alpha = std_s1 / std_s2


# POS pulse signal
pulse_signal = s1 + alpha * s2


print(f"POS alpha: {alpha:.4f}")


# --------------------------------------------------
# 7. Band-pass filter
# --------------------------------------------------

low_cutoff = 0.7
high_cutoff = 4.0

nyquist = fps / 2

low = low_cutoff / nyquist
high = high_cutoff / nyquist

if high >= 1:
    high = 0.99

b, a = butter(
    3,
    [low, high],
    btype="bandpass"
)

filtered_signal = filtfilt(
    b,
    a,
    pulse_signal
)


# --------------------------------------------------
# 8. FFT
# --------------------------------------------------

n = len(filtered_signal)

frequencies = np.fft.rfftfreq(
    n,
    d=1 / fps
)

fft_values = np.abs(
    np.fft.rfft(filtered_signal)
)


# Only consider valid heart-rate frequencies
valid = (
    (frequencies >= low_cutoff)
    &
    (frequencies <= high_cutoff)
)

valid_frequencies = frequencies[valid]
valid_fft = fft_values[valid]


if len(valid_fft) == 0:
    print("ERROR: Could not find valid heart-rate frequency.")
    exit()


# Find strongest frequency
peak_index = np.argmax(valid_fft)

dominant_frequency = valid_frequencies[peak_index]


# --------------------------------------------------
# 9. Convert frequency to BPM
# --------------------------------------------------

bpm = dominant_frequency * 60


print()
print("--------------------------------")
print("HEART RATE RESULT")
print("--------------------------------")
print(f"Dominant frequency: {dominant_frequency:.3f} Hz")
print(f"Estimated Heart Rate: {bpm:.1f} BPM")
print("--------------------------------")


# --------------------------------------------------
# 10. Plot rPPG signal
# --------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    time,
    filtered_signal
)

plt.xlabel("Time (seconds)")
plt.ylabel("rPPG Signal")
plt.title("Filtered rPPG Pulse Signal")

plt.grid()

plt.show()


# --------------------------------------------------
# 11. Plot FFT
# --------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    valid_frequencies,
    valid_fft
)

plt.axvline(
    dominant_frequency,
    linestyle="--",
    label=f"Peak = {dominant_frequency:.2f} Hz"
)

plt.xlabel("Frequency (Hz)")
plt.ylabel("FFT Magnitude")
plt.title("FFT of rPPG Signal")

plt.legend()
plt.grid()

plt.show()