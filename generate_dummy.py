from PIL import Image
import numpy as np
import os

os.makedirs("data/rgb", exist_ok=True)

for i in range(1, 4):
    arr = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    img.save(f"data/rgb/sample{i}.png")
