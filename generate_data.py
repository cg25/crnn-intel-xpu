import cv2
import numpy as np

for i in range(100):
    text = f"test{i:03d}"
    img = np.ones((32, 128), dtype=np.uint8) * 255

    font = cv2.FONT_HERSHEY_SIMPLEX
    pos = (5, 25)
    color = 0
    thickness = 1

    cv2.putText(img, text, pos, font, 0.8, color, thickness)

    noise = np.random.randint(0, 30, (32, 128), dtype=np.uint8)
    img = cv2.add(img, noise)

    cv2.imwrite(f"data/synthetic/{text}.png", img)