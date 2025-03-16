import cv2
import numpy as np
import random
import os

os.makedirs("data/train", exist_ok=True)
os.makedirs("data/validation", exist_ok=True)
os.makedirs("data/test", exist_ok=True)

charset = 'abcdefghijklmnopqrstuvwxyz0123456789'

def generate_image(path):
    length = 5

    label = ''.join(random.choice(charset) for _ in range(length))

    base_height = 32
    img = np.ones((base_height, 128), dtype=np.uint8) * 255

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 1

    (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
    while text_w > 120 and font_scale > 0.5:
        font_scale -= 0.1
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

    x = (128 - text_w) // 2
    y = (base_height + text_h) // 2

    cv2.putText(img, label, (x, y), font, font_scale, 0, thickness)

    cv2.imwrite(f"{path}{label}.png", img)

for _ in range(1000):
    generate_image("data/train/")

for _ in range(200):
    generate_image("data/validation/")

for _ in range(5):
    generate_image("data/test/")