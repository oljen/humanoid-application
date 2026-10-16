import cv2, glob, os, numpy as np
LEVELS = {"motion25": 25, "motion40": 40, "motion60": 60}
rng = np.random.default_rng(1)

def motion_kernel(length, angle):
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1
    rot = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1)
    k = cv2.warpAffine(k, rot, (length, length))
    return k / k.sum()

for clip in sorted(glob.glob("my_clips/good/*")):
    base = os.path.splitext(os.path.basename(clip))[0]
    angle = rng.uniform(0, 180)
    for group, length in LEVELS.items():
        os.makedirs(f"my_clips/{group}", exist_ok=True)
        out = f"my_clips/{group}/{base}.mp4"
        if os.path.exists(out):
            continue
        cap = cv2.VideoCapture(clip)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        k, writer = motion_kernel(length, angle), None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.filter2D(frame, -1, k)
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(frame)
        cap.release(); writer.release()
    print("done", base)
