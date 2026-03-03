from deepface import DeepFace
import cv2, os
from Foundation import NSURL
from AVFoundation import AVAudioPlayer
import time
import numpy as np

detector_backend = "opencv"  # or "mediapipe"; use "retinaface" only if installed
CONF_THRESH = 0.3
BOOM_PATH = "vine-boom.mp3"
STEP = 7
last_label = "neutral"
last_conf = CONF_THRESH
frame_idx = 0

# load audio boom
url = NSURL.fileURLWithPath_(BOOM_PATH)
player, err = AVAudioPlayer.alloc().initWithContentsOfURL_error_(url, None)
assert player is not None, err
player.prepareToPlay()  # prebuffer for instant start

def play_boom():
    player.stop()            # rewind if it’s mid-play
    player.setCurrentTime_(0)
    player.play()


# match the image
def match_image(label):
    if not label: return None
    p = os.path.join("expressions", f"{label}.jpeg")  # use 'angry.jpg' (not 'anger.jpg')
    return p if os.path.isfile(p) else None

cap = cv2.VideoCapture(1)  # try 0 instead of 1 if webcam fails
while True:
    ok, frame = cap.read()
    if not ok: break

    frame_idx += 1

    label, conf = None, 0.0

    res = DeepFace.analyze(
        img_path=frame,
        actions=['emotion'],
        detector_backend=detector_backend,
        enforce_detection=False,
    )
    item  = res[0] if isinstance(res, list) else res

    if frame_idx % STEP == 0:
        label = item["dominant_emotion"]                      # angry/disgust/fear/happy/sad/surprise/neutral
        conf  = float(item["emotion"][label]) / 100.0
        if label != last_label:
            play_boom()
        last_label = label
        last_conf = conf
        frame_idx = 0
    else:
        label = last_label
        conf = last_conf
        
    cv2.putText(frame, f"{label or '—'}  {conf:.2f}", (12,32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    POP_SECS = 2.0

    # Persistent state (define once before your loop)
    try:
        _overlay_expire
    except NameError:
        _overlay_expire = 0.0
        _overlay_path = None

    # --- inside your capture loop ---
    now = time.monotonic()

    mp = match_image(label) if (label and conf >= CONF_THRESH) else None

    # Arm/rearm the popup if a new image appears or previous one expired
    if mp and (mp != _overlay_path or now >= _overlay_expire):
        _overlay_path = mp
        _overlay_expire = now + POP_SECS

    # Blend only while the timer is active
    if now < _overlay_expire and _overlay_path:
        ov = cv2.imread(_overlay_path, cv2.IMREAD_UNCHANGED)
        if ov is not None:
            H, W = frame.shape[:2]
            oh, ow = ov.shape[:2]

            # Scale to fit window (keep aspect)
            s = min(W / ow, H / oh)
            nw, nh = max(1, int(ow * s)), max(1, int(oh * s))
            ov = cv2.resize(ov, (nw, nh), interpolation=cv2.INTER_AREA)

            # Split color + alpha (support PNG alpha or uniform alpha)
            if ov.ndim == 3 and ov.shape[2] == 4:
                ov_bgr = ov[:, :, :3].astype(np.float32)
                ov_a   = (ov[:, :, 3].astype(np.float32) / 255.0) * 0.3
            else:
                ov_bgr = ov.astype(np.float32)
                ov_a   = np.full((nh, nw), 0.3, dtype=np.float32)

            # Center the overlay
            y0 = (H - nh) // 2; x0 = (W - nw) // 2
            y1, x1 = y0 + nh, x0 + nw

            # Safety crop
            yy0, xx0 = max(0, y0), max(0, x0)
            yy1, xx1 = min(H, y1), min(W, x1)
            if yy1 > yy0 and xx1 > xx0:
                roi = frame[yy0:yy1, xx0:xx1].astype(np.float32)
                oy0, ox0 = yy0 - y0, xx0 - x0
                ov_crop  = ov_bgr[oy0:y1 - y0, ox0:x1 - x0]
                a_crop   = ov_a[oy0:y1 - y0, ox0:x1 - x0][..., None]
                out = a_crop * ov_crop + (1.0 - a_crop) * roi
                frame[yy0:yy1, xx0:xx1] = out.astype(np.uint8)

    # Show frame as usual
    cv2.imshow("DeepFace emotion matcher", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
