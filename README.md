# Real-Time Face-Based rPPG Heart Rate Monitor 

# ( "STILL IN DEVELOPING" )

## Overview
This project implements a real-time heart rate monitor using a standard webcam and computer vision techniques.  
It uses **remote photoplethysmography (rPPG)** to measure subtle color changes in the forehead region caused by blood flow, allowing non-contact heart rate detection.

A live waveform plot of the heartbeat signal is displayed alongside the real-time video feed.

---

## Features
- Detects heart rate from the **forehead region** using Mediapipe's Face Mesh.
- **Non-contact** — no wearable sensors required.
- Real-time BPM calculation.
- Live waveform plot showing raw and filtered signals.
- Adjustable **bandpass filter** for accurate pulse extraction.
- Works with most standard webcams.

---

## How It Works
1. **Face Detection:** Uses Mediapipe Face Mesh to find facial landmarks.
2. **ROI Extraction:** Selects a stable forehead region for analysis.
3. **Signal Processing:**  
   - Extracts the mean green channel intensity from the ROI.  
   - Detrends the signal to remove lighting variation.  
   - Applies a bandpass filter to isolate heartbeat frequencies.
4. **BPM Calculation:** Detects peaks in the filtered signal and converts the average interval to beats per minute.
5. **Visualization:** Displays both the video feed with BPM overlay and a real-time waveform plot.

---

## Installation

### Requirements
Install Python dependencies:
```bash
pip install opencv-python mediapipe numpy scipy matplotlib
```

---

## Usage
Run the script:
```bash
python face_heartbeat_with_plot.py
```

Controls:
- Press **q** to quit.

---

## Parameters
You can adjust these parameters in the script:
- `DESIRED_FPS` — target frame rate for capture.
- `BUFFER_SECONDS` — signal history length for analysis.
- `LOW_CUT` / `HIGH_CUT` — bandpass filter frequency range in Hz.

---

## Limitations
- Requires **good lighting** for accurate detection.
- Works best when the subject is **still** and facing the camera.
- Not a medical device — intended for **research and educational purposes** only.

---
