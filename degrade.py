import cv2, glob, os, sys

LEVELS = {"gauss6": 6, "gauss8": 8, "gauss10": 10, "gauss12": 12, "gauss24": 24}
only_check = "--check" in sys.argv

clips = sorted(glob.glob("my_clips/good/*"))
if only_check:   # just save one frame per level to look at
    cap = cv2.VideoCapture(clips[4]); cap.set(cv2.CAP_PROP_POS_FRAMES, 150)
    ok, frame = cap.read()
    row = [frame] + [cv2.GaussianBlur(frame, (0, 0), s) for s in LEVELS.values()]
    small = [cv2.resize(f, (640, 360)) for f in row]
    for f, label in zip(small, ["original"] + list(LEVELS)):
        cv2.putText(f, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imwrite("blur_check.jpg", cv2.vconcat([cv2.hconcat(small[:2]), cv2.hconcat(small[2:])]))
    print("saved blur_check.jpg")
    sys.exit()

for clip in clips:
    base = os.path.splitext(os.path.basename(clip))[0]
    for group, sigma in LEVELS.items():
        os.makedirs(f"my_clips/{group}", exist_ok=True)
        out = f"my_clips/{group}/{base}.mp4"
        if os.path.exists(out):
            continue
        cap = cv2.VideoCapture(clip)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        writer = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.GaussianBlur(frame, (0, 0), sigma)
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(frame)
        cap.release(); writer.release()
    print("done", base)
