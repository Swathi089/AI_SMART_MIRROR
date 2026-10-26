import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.signal import butter, sosfiltfilt, welch, find_peaks


# ==================================================
# 1. Load and validate the existing recording
# ==================================================
CSV_PATH = "modules/heart_rate/rgb_data.csv"

data = pd.read_csv(CSV_PATH)

required_columns = ["time", "red", "green", "blue"]

if not all(column in data.columns for column in required_columns):
    raise ValueError("CSV must contain time, red, green, blue columns.")

data = data[required_columns].apply(pd.to_numeric, errors="coerce")
data = data.replace([np.inf, -np.inf], np.nan).dropna()
data = data.sort_values("time")
data = data.drop_duplicates(subset="time")

time = data["time"].to_numpy(dtype=float)
rgb = data[["red", "green", "blue"]].to_numpy(dtype=float)

if len(time) < 300:
    raise ValueError("Not enough valid samples. Record for at least 20 seconds.")

if np.any(np.diff(time) <= 0):
    raise ValueError("Timestamps must increase.")

duration = time[-1] - time[0]

if duration < 20:
    raise ValueError("Recording must be at least 20 seconds long.")

print("Samples:", len(time))
print(f"Duration: {duration:.2f} seconds")


# ==================================================
# 2. Resample irregular timestamps uniformly
# ==================================================
intervals = np.diff(time)
fps = 1.0 / np.median(intervals)

# Use the observed frame rate, capped to avoid oversampling.
fps = min(fps, 60.0)

uniform_time = np.arange(time[0], time[-1], 1.0 / fps)

rgb_uniform = np.column_stack([
    interp1d(
        time,
        rgb[:, channel],
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate"
    )(uniform_time)
    for channel in range(3)
])

print(f"Resampled rate: {fps:.2f} FPS")


# ==================================================
# 3. Extract the POS pulse signal
# ==================================================
# POS uses short overlapping windows to reduce the
# influence of gradual brightness changes.
window_seconds = 1.6
window_size = int(round(window_seconds * fps))
step_size = max(1, window_size // 2)

pulse_sum = np.zeros(len(uniform_time))
weight_sum = np.zeros(len(uniform_time))

weights = np.hanning(window_size)

for start in range(0, len(uniform_time) - window_size + 1, step_size):
    end = start + window_size
    window_rgb = rgb_uniform[start:end].copy()

    means = np.mean(window_rgb, axis=0)

    if np.any(means <= 1e-8):
        continue

    normalized = window_rgb / means
    R, G, B = normalized.T

    s1 = G - B
    s2 = G + B - 2.0 * R

    std_s2 = np.std(s2)

    if std_s2 < 1e-8:
        continue

    alpha = np.std(s1) / std_s2
    h = s1 + alpha * s2
    h = h - np.mean(h)

    pulse_sum[start:end] += h * weights
    weight_sum[start:end] += weights

valid = weight_sum > 1e-8

pulse = np.zeros_like(pulse_sum)
pulse[valid] = pulse_sum[valid] / weight_sum[valid]

# Keep only the continuous region with a usable signal.
valid_indices = np.flatnonzero(valid)

if len(valid_indices) < int(20 * fps):
    raise ValueError("Not enough usable POS signal.")

first = valid_indices[0]
last = valid_indices[-1] + 1

pulse = pulse[first:last]
signal_time = uniform_time[first:last]

# Remove any remaining slow drift.
pulse = pulse - np.mean(pulse)


# ==================================================
# 4. Bandpass filter: 42–180 BPM
# ==================================================
low_hz = 42.0 / 60.0
high_hz = 180.0 / 60.0

if high_hz >= fps / 2:
    high_hz = 0.45 * fps

sos = butter(
    3,
    [low_hz, high_hz],
    btype="bandpass",
    fs=fps,
    output="sos"
)

if len(pulse) < 30:
    raise ValueError("Signal is too short to filter.")

filtered = sosfiltfilt(sos, pulse)


# ==================================================
# 5. Analyse overlapping 12-second segments
# ==================================================
segment_seconds = 12.0
segment_size = int(round(segment_seconds * fps))
segment_step = segment_size // 2

if len(filtered) < segment_size:
    raise ValueError("Need at least 12 seconds of usable signal.")

segment_results = []
all_candidates = []

def get_candidates(segment):
    segment = segment - np.mean(segment)

    frequencies, power = welch(
        segment,
        fs=fps,
        window="hann",
        nperseg=len(segment),
        noverlap=0,
        nfft=16384,
        detrend="constant"
    )

    mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    f = frequencies[mask]
    p = power[mask]

    if len(p) < 3 or not np.any(p > 0):
        return f, p, []

    peaks, _ = find_peaks(p)

    if len(peaks) == 0:
        peaks = np.array([int(np.argmax(p))])

    ranked = peaks[np.argsort(p[peaks])[::-1]]

    candidates = [
        {
            "bpm": float(f[i] * 60.0),
            "power": float(p[i]),
            "relative_power": float(p[i] / (np.max(p) + 1e-12))
        }
        for i in ranked[:3]
    ]

    return f, p, candidates


print("\n========== SEGMENT RESULTS ==========")

for start in range(0, len(filtered) - segment_size + 1, segment_step):
    end = start + segment_size
    segment = filtered[start:end]

    f, p, candidates = get_candidates(segment)

    if not candidates:
        continue

    best = candidates[0]
    segment_results.append(best["bpm"])
    all_candidates.append(candidates)

    print(
        f"{signal_time[start] - signal_time[0]:5.1f}–"
        f"{signal_time[end - 1] - signal_time[0]:5.1f} sec | "
        f"Top: {best['bpm']:.1f} BPM"
    )

    print(
        "    Other candidates:",
        ", ".join(f"{c['bpm']:.1f}" for c in candidates[1:])
        if len(candidates) > 1 else "none"
    )

if not segment_results:
    raise ValueError("No usable segment estimates found.")


# ==================================================
# 6. Summarize consistency without forcing a result
# ==================================================
segment_results = np.array(segment_results)

median_bpm = float(np.median(segment_results))
spread_bpm = float(np.percentile(segment_results, 75)
                   - np.percentile(segment_results, 25))

print("\n========== SUMMARY ==========")
print("Segment estimates:", np.round(segment_results, 1).tolist())
print(f"Median candidate: {median_bpm:.1f} BPM")
print(f"Interquartile spread: {spread_bpm:.1f} BPM")

if len(segment_results) >= 2 and spread_bpm <= 10:
    print("Consistency: segment estimates are reasonably close.")
else:
    print("Consistency: estimates vary; result may be unreliable.")

print(
    "IMPORTANT: consistency is not proof of accuracy. "
    "Compare with a pulse oximeter measured at the same time."
)


# ==================================================
# 7. Plot filtered signal
# ==================================================
plt.figure(figsize=(11, 4))
plt.plot(signal_time - signal_time[0], filtered)
plt.xlabel("Time (seconds)")
plt.ylabel("Filtered POS signal")
plt.title("Filtered Webcam Pulse Signal")
plt.grid(True)
plt.tight_layout()


# ==================================================
# 8. Plot the spectrum of the full recording
# ==================================================
frequencies, power = welch(
    filtered,
    fs=fps,
    window="hann",
    nperseg=min(len(filtered), int(12 * fps)),
    noverlap=min(len(filtered), int(12 * fps)) // 2,
    nfft=16384,
    detrend="constant"
)

mask = (frequencies >= low_hz) & (frequencies <= high_hz)

plt.figure(figsize=(11, 4))
plt.plot(frequencies[mask] * 60, power[mask])
plt.axvline(
    median_bpm,
    linestyle="--",
    label=f"Median segment candidate: {median_bpm:.1f} BPM"
)
plt.xlabel("Frequency (BPM)")
plt.ylabel("Power")
plt.title("Webcam Pulse Frequency Spectrum")
plt.xlim(42, 180)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()