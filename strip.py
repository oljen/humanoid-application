import sys, imageio, numpy as np
for name in sys.argv[1:]:
    frames = imageio.mimread(f"retargeted/{name}.mp4", memtest=False)
    pick = np.linspace(0, len(frames) - 1, 10).astype(int)
    big = [np.kron(frames[i], np.ones((2, 2, 1), dtype=np.uint8)) for i in pick]
    imageio.imwrite(f"retargeted/{name}_strip.png", np.hstack(big))
    print("saved", f"retargeted/{name}_strip.png")
