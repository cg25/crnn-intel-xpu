import cv2
import numpy as np
from pathlib import Path
import openvino as ov


class CRNNOpenVINOInference:
    def __init__(self, model_path: str, charset_path: str, img_h: int = 32):
        self.core = ov.Core()

        self.compiled_model = self.core.compile_model(
            model=model_path,
            device_name="GPU"
        )
        self.infer_request = self.compiled_model.create_infer_request()

        with open(charset_path, 'r', encoding='utf-8-sig') as f:
            self.charset = [line.strip() for line in f if line.strip()]

        self.img_h = img_h
        self.idx_to_char = {idx: char for idx, char in enumerate(self.charset)}

    def preprocess(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        h, w = image.shape
        new_w = int(w * (self.img_h / h))
        resized_img = cv2.resize(image, (new_w, self.img_h))

        normalized = resized_img.astype(np.float32) / 255.0
        return np.expand_dims(normalized, axis=(0, 1))  # [1,1,H,W]

    def decode_ctc(self, output: np.ndarray) -> str:
        seq_len, _, num_classes = output.shape
        indices = np.argmax(output, axis=2).flatten()

        decoded = []
        prev_idx = -1
        for idx in indices:
            if idx == 0:
                continue
            if idx != prev_idx:
                decoded.append(self.idx_to_char[idx])
            prev_idx = idx
        return ''.join(decoded)

    def predict(self, image_path: str) -> str:
        input_data = self.preprocess(image_path)

        input_tensor = ov.Tensor(array=input_data, shared_memory=True)
        self.infer_request.set_input_tensor(input_tensor)

        self.infer_request.start_async()
        self.infer_request.wait()

        output = self.infer_request.get_output_tensor().data
        return self.decode_ctc(output)


if __name__ == "__main__":
    MODEL_DIR = "../outputs/crnn.onnx"
    CHARSET_PATH = "../data/charset.txt"
    IMG_H = 32

    recognizer = CRNNOpenVINOInference(
        model_path=MODEL_DIR,
        charset_path=CHARSET_PATH,
        img_h=IMG_H
    )

    test_images = list(Path("../data/test").glob("*.[pj][np]g"))

    for img_path in test_images:
        if img_path.exists():
            try:
                pred = recognizer.predict(str(img_path))
                print(f"input：{img_path.name} → predict：{pred}")
            except Exception as e:
                print(f"handling {img_path.name} got error：{str(e)}")
        else:
            print(f"image not found：{img_path}")