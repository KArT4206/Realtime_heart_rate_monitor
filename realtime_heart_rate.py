"""
Real-time face-based rPPG heart rate monitor with live waveform plot.

Requirements:
    pip install opencv-python mediapipe numpy scipy matplotlib

Run:
    python face_heartbeat_with_plot.py

Press 'q' in the OpenCV window to quit.
"""
import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt

# ----- PARAMETERS -----
DESIRED_FPS = 30
BUFFER_SECONDS = 12
BUFFER_LEN = int(DESIRED_FPS * BUFFER_SECONDS)
LOW_CUT = 0.75   # Hz (~45 BPM)
HIGH_CUT = 3.0   # Hz (~180 BPM)
FILTER_ORDER = 4

# Stable forehead region
FOREHEAD_POINTS = [10, 338, 297, 332, 284, 251, 389, 356]


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 1e-6)
    high = min(highcut / nyq, 0.9999)
    b, a = butter(order, [low, high], btype="band")
    return b, a


def bandpass_filtfilt(signal, fs, lowcut=LOW_CUT, highcut=HIGH_CUT, order=FILTER_ORDER):
    # If not enough samples for filtfilt, return original array
    if len(signal) < (order * 3):
        return np.array(signal)
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    try:
        y = filtfilt(b, a, signal)
    except Exception:
        # fallback to lfilter if filtfilt errors (less zero-phase but safer)
        from scipy.signal import lfilter
        y = lfilter(b, a, signal)
    return y


def compute_bpm_from_filtered(filtered_signal, times, fps_int):
    if len(filtered_signal) < max(3, int(0.5 * fps_int)):
        return None
    min_distance = max(1, int(0.4 * fps_int))  # avoid double peaks
    peaks, _ = find_peaks(filtered_signal, distance=min_distance)
    if len(peaks) < 2:
        return None
    peak_times = np.array([times[p] for p in peaks])
    intervals = np.diff(peak_times)
    intervals = intervals[intervals > 0]
    if len(intervals) == 0:
        return None
    mean_interval = np.mean(intervals)
    if mean_interval <= 0:
        return None
    bpm = 60.0 / mean_interval
    return bpm, peaks


# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Camera setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open camera (index 0)")

cap.set(cv2.CAP_PROP_FPS, DESIRED_FPS)
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Failed to read frame from camera.")
frame_h, frame_w = frame.shape[:2]

camera_fps = cap.get(cv2.CAP_PROP_FPS)
fps = camera_fps if camera_fps and camera_fps > 1.0 else DESIRED_FPS
fps_int = max(1, int(round(fps)))  # integer sample-rate for window sizes
print(f"[INFO] Camera FPS used for processing: {fps:.2f} (fps_int={fps_int})")

# Buffers
signal_buffer = deque(maxlen=BUFFER_LEN)   # raw green channel signal
time_buffer = deque(maxlen=BUFFER_LEN)     # timestamps for each sample
bpm_buffer = deque(maxlen=8)               # smoother display

# Matplotlib live plot
plt.ion()
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.set_xlim(0, BUFFER_SECONDS)
ax.set_ylim(-40, 40)
ax.set_xlabel("seconds (latest → right)")
ax.set_ylabel("signal (relative)")
plt.tight_layout()

last_plot_time = time.time()
print("Press 'q' in the video window to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = time.time()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        forehead_roi = None
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            pts = []
            for idx in FOREHEAD_POINTS:
                lm = face_landmarks.landmark[idx]
                x, y = int(lm.x * frame_w), int(lm.y * frame_h)
                pts.append((x, y))

            x_coords = [p[0] for p in pts]
            y_coords = [p[1] for p in pts]
            x_min, x_max = max(min(x_coords) - 3, 0), min(max(x_coords) + 3, frame_w - 1)
            y_min, y_max = max(min(y_coords) - 3, 0), min(max(y_coords) + 3, frame_h - 1)

            if x_max > x_min and y_max > y_min:
                forehead_roi = frame[y_min:y_max, x_min:x_max]

            cv2.polylines(frame, [np.array(pts, np.int32)], True, (0, 255, 0), 1)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 200, 0), 1)

        # Extract mean green from ROI; keep time buffer aligned
        if forehead_roi is not None and forehead_roi.size > 0:
            green_mean = float(np.mean(forehead_roi[:, :, 1]))
            signal_buffer.append(green_mean)
            time_buffer.append(t)
        else:
            # if no ROI, repeat last sample's value (if exists) to keep timing; append timestamp
            if len(signal_buffer) > 0:
                signal_buffer.append(signal_buffer[-1])
            else:
                signal_buffer.append(0.0)
            time_buffer.append(t)

        # Minimum safe samples to start processing: 2 * fps_int (2 seconds)
        if len(signal_buffer) >= int(2 * fps_int):
            sig = np.array(signal_buffer)
            times = np.array(time_buffer)

            # Detrend: moving-average window of ~1 second (use fps_int samples)
            window = max(1, int(fps_int))
            if len(sig) >= window:
                # convolve returns same size with mode='same'
                ma = np.convolve(sig, np.ones(window) / window, mode="same")
                # Ensure ma length equals sig length
                if ma.shape[0] == sig.shape[0]:
                    sig_detrended = sig - ma
                else:
                    # fallback safe detrend
                    sig_detrended = sig - np.mean(sig)
            else:
                sig_detrended = sig - np.mean(sig)

            # Bandpass filter (returns array same length as input)
            filtered = bandpass_filtfilt(sig_detrended, fs=fps, lowcut=LOW_CUT, highcut=HIGH_CUT)
            # Make sure filtered length matches sig length
            if filtered.shape[0] != sig.shape[0]:
                # If mismatch, pad/truncate safely
                n = sig.shape[0]
                filtered = np.resize(filtered, n)

            # Compute BPM using filtered & times
            bpm_result = compute_bpm_from_filtered(filtered, times, fps_int)
            if bpm_result is not None:
                bpm_val, _ = bpm_result
                if 30 < bpm_val < 220:
                    bpm_buffer.append(bpm_val)

            bpm_display = int(np.round(np.mean(bpm_buffer))) if bpm_buffer else 0
        else:
            # not enough data yet
            filtered = np.zeros(len(signal_buffer))
            bpm_display = 0

        # Overlay BPM
        cv2.putText(frame, f"Heart Rate: {bpm_display} BPM", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imshow("rPPG Heart Rate (press q to quit)", frame)

        # Plot update ~4x/sec (keep CPU usage reasonable)
        if time.time() - last_plot_time > 0.25 and len(time_buffer) > 2:
            last_plot_time = time.time()
            tb = np.array(time_buffer)
            tb_rel = tb - tb[-1]
            sig_arr = np.array(signal_buffer) - np.mean(signal_buffer)
            filt_vis = np.array(filtered) if (len(filtered) == len(sig_arr)) else np.zeros_like(sig_arr)

            ax.clear()
            ax.plot(-tb_rel, sig_arr, linewidth=0.9, label="raw (green)")
            ax.plot(-tb_rel, filt_vis, linewidth=1.1, label="filtered")
            ax.set_xlim(0, BUFFER_SECONDS)
            ystd = max(1e-3, np.std(sig_arr))
            ax.set_ylim(-6 * ystd, 6 * ystd)
            ax.set_xlabel("seconds (latest → right)")
            ax.set_ylabel("relative signal")
            ax.set_title(f"Real-time waveform — BPM: {bpm_display}")
            ax.legend(loc="upper right")
            plt.pause(0.001)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.close(fig)
    face_mesh.close()
    print("Exiting.")
