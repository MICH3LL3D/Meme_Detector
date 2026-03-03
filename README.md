## Meme Detector (`face_match_test.py`)

This is a simple webcam-based meme detector that:
- **Analyzes your facial expression** in real time using DeepFace.
- **Displays the dominant emotion + confidence** on the video feed.
- **Shows a matching meme image** on screen (for a short popup) based on the detected emotion.
- **Plays a “vine boom” sound effect** whenever the detected emotion changes.

### Requirements

- **Python 3.9+**
- **Packages**:
  - `deepface`
  - `opencv-python`
  - `numpy`
  - `pyobjc` (for `Foundation` and `AVFoundation` on macOS)
- **Assets**:
  - `vine-boom.mp3` in the project root.
  - An `expressions/` folder with images named after emotions, e.g.:
    - `expressions/happy.jpeg`
    - `expressions/sad.jpeg`
    - `expressions/angry.jpeg`
    - etc. (filenames must match the emotion labels used by DeepFace).

Install dependencies (example):

```bash
pip install deepface opencv-python numpy pyobjc
```

### How to Run

From the project directory:

```bash
python face_match_test.py
```

- If your webcam does not appear, change the camera index in `face_match_test.py`:
  - Line with `cv2.VideoCapture(1)` → try `cv2.VideoCapture(0)` instead.

### How It Works (Brief)

- Captures frames from your webcam and runs `DeepFace.analyze(..., actions=['emotion'])`.
- Every few frames, it:
  - Reads the **dominant emotion** and its **confidence**.
  - If the emotion changed since the last update, plays `vine-boom.mp3`.
  - Looks for a matching image in `expressions/<emotion>.jpeg` and blends it over the video for a short time.
  - Draws `"<emotion>  <confidence>"` text on the video window titled `DeepFace emotion matcher`.
