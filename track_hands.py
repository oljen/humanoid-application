import sys, os, glob
import numpy as np, cv2, mediapipe as mp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

args = [a for a in sys.argv[1:] if not a.startswith("--")]
overlay = "--overlay" in sys.argv
clips = args or sorted(glob.glob("my_clips/*/*"))
clips = [c for c in clips if c.lower().endswith((".mov", ".mp4"))]
os.makedirs("tracks", exist_ok=True)

mp_hands = mp.solutions.hands
draw = mp.solutions.drawing_utils

for clip in clips:
    group = os.path.basename(os.path.dirname(clip))          # good / hard / fail
    name = group + "_" + os.path.splitext(os.path.basename(clip))[0]
    cap = cv2.VideoCapture(clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    writer = None
    lms, scores = [], []
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.multi_hand_landmarks:
                hl = res.multi_hand_landmarks[0]
                lms.append([[p.x, p.y, p.z] for p in hl.landmark])
                scores.append(res.multi_handedness[0].classification[0].score)
                if overlay:
                    draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
            else:
                lms.append(np.full((21, 3), np.nan))
                scores.append(0.0)
            if overlay:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(f"tracks/{name}_overlay.mp4",
                                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                writer.write(frame)
    cap.release()
    if writer:
        writer.release()

    lms = np.array(lms, dtype=float)      # (frames, 21 points, xyz)
    scores = np.array(scores)
    np.savez(f"tracks/{name}.npz", landmarks=lms, score=scores, fps=fps)

    # Pinch = thumb tip to index tip, divided by hand size (wrist to middle knuckle)
    # so it doesn't change when the hand moves closer to or further from the camera.
    pinch = (np.linalg.norm(lms[:, 4, :2] - lms[:, 8, :2], axis=1) /
             np.linalg.norm(lms[:, 0, :2] - lms[:, 9, :2], axis=1))
    t = np.arange(len(lms)) / fps

    fig, ax = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    ax[0].plot(t, lms[:, 0, 0], label="wrist x")
    ax[0].plot(t, lms[:, 0, 1], label="wrist y")
    ax[0].set_ylabel("image position (0-1)"); ax[0].legend()
    ax[1].plot(t, pinch); ax[1].set_ylabel("pinch (thumb-index / hand size)")
    ax[2].plot(t, scores); ax[2].set_ylabel("detector score"); ax[2].set_xlabel("seconds")
    ax[0].set_title(name)
    fig.tight_layout(); fig.savefig(f"tracks/{name}.png"); plt.close(fig)

    found = np.mean(scores > 0) * 100
    print(f"{name}: {len(lms)} frames, hand found in {found:.0f}% of frames")
