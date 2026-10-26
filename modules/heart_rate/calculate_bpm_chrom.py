import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, welch, find_peaks


# ============================================================
# LOAD DATA
# ============================================================

data = pd.read_csv("modules/heart_rate/rgb_data.csv")

time = data["time"].values
rgb = data[["red", "green", "blue"]].values

print("Total samples:", len(data))


# ============================================================
# REMOVE FIRST 8 SECONDS
# ============================================================

relative_time = time - time[0]

mask = relative_time >= 8

time = time[mask]
rgb = rgb[mask]

print("Samples after removing first 8 seconds:", len(rgb))


# ============================================================
# ESTIMATE FPS
# ============================================================

fps = 1 / np.mean(np.diff(time))

print(f"Estimated FPS: {fps:.2f}")


# ============================================================
# NORMALIZE RGB
# ============================================================

rgb_mean = np.mean(rgb, axis=0)

rgb_normalized = rgb / rgb_mean

R = rgb_normalized[:, 0]
G = rgb_normalized[:, 1]
B = rgb_normalized[:, 2]


# ============================================================
# CHROM METHOD
# ============================================================

# Chrominance signals
X = 3 * R - 2 * G
Y = 1.5 * R + G - 1.5 * B


# Standard deviation ratio
std_X = np.std(X)
std_Y = np.std(Y)

alpha = std_X / (std_Y + 1e-8)

chrom_signal = X - alpha * Y


# ============================================================
# BANDPASS FILTER
# ============================================================

low = 0.7
high = 3.0

b, a = butter(
    3,
    [low / (fps / 2), high / (fps / 2)],
    btype="band"
)

filtered = filtfilt(b, a, chrom_signal)


# ============================================================
# WELCH POWER SPECTRUM
# ============================================================

frequencies, power = welch(
    filtered,
    fs=fps,
    window="hann",
    nperseg=min(512, len(filtered)),
    noverlap=None,
    nfft=4096
)


# ============================================================
# HEART RATE RANGE
# 45 - 120 BPM
# ============================================================

valid = (
    (frequencies >= 0.75) &
    (frequencies <= 2.0)
)

freq = frequencies[valid]
psd = power[valid]


# ============================================================
# FIND PEAKS
# ============================================================

peaks, properties = find_peaks(
    psd,
    distance=10,
    prominence=np.max(psd) * 0.02
)

if len(peaks) == 0:
    print("No reliable peak detected.")
    exit()


# ============================================================
# SORT PEAKS BY POWER
# ============================================================

sorted_peaks = peaks[
    np.argsort(psd[peaks])[::-1]
]


print()
print("--------------------------------")
print("CHROM TOP FREQUENCY PEAKS")
print("--------------------------------")

for i, p in enumerate(sorted_peaks[:10]):

    frequency = freq[p]
    bpm = frequency * 60

    print(
        f"{i+1}. "
        f"{frequency:.3f} Hz = "
        f"{bpm:.1f} BPM "
        f"| Power = {psd[p]:.6f}"
    )


# ============================================================
# SELECT STRONGEST CHROM PEAK
# ============================================================

best_peak = sorted_peaks[0]

dominant_frequency = freq[best_peak]

bpm = dominant_frequency * 60


# ============================================================
# QUALITY
# ============================================================

peak_power = psd[best_peak]

median_power = np.median(psd)

quality = peak_power / (median_power + 1e-8)


# ============================================================
# RESULT
# ============================================================

print()
print("--------------------------------")
print("CHROM HEART RATE RESULT")
print("--------------------------------")

print(
    f"Dominant frequency: "
    f"{dominant_frequency:.3f} Hz"
)

print(
    f"Estimated Heart Rate: "
    f"{bpm:.1f} BPM"
)

print(
    f"Peak Quality Ratio: "
    f"{quality:.2f}"
)

print("--------------------------------")


# ============================================================
# PLOT CHROM SIGNAL
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(time, filtered)

plt.xlabel("Time (seconds)")
plt.ylabel("CHROM rPPG Signal")

plt.title("CHROM - Filtered rPPG Signal")

plt.grid()

plt.show()


# ============================================================
# PLOT WELCH SPECTRUM
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(freq, psd)

plt.axvline(
    dominant_frequency,
    linestyle="--",
    label=f"Peak = {dominant_frequency:.3f} Hz"
)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")

plt.title("CHROM Welch Power Spectrum")

plt.legend()

plt.grid()

plt.show()