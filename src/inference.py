import torch
import cv2
import numpy as np
from pathlib import Path
from core.crnn import CRNN


class CRNNInference:
    def __init__(self, checkpoint_path: str, charset_path: str, img_h: int = 32, device: str = "xpu"):
        self.device = torch.device(device)
        assert torch.xpu.is_available(), "XPU设备不可用"

        with open(charset_path, 'r', encoding='utf-8-sig') as f:
            self.charset = f.read().splitlines()


        self.model = CRNN(img_h=img_h, num_classes=len(self.charset)).to(self.device)
        self._load_checkpoint(checkpoint_path)
        self.model.eval()

        self.idx_to_char = {idx: char for idx, char in enumerate(self.charset)}
        self.char_to_idx = {char: idx for idx, char in self.idx_to_char.items()}

    def _load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        print(f"成功加载检查点：{Path(path).name}")

    def preprocess(self, image_path: str) -> torch.Tensor:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        h, w = image.shape
        new_w = int(w * (self.model.img_h / h))
        resized_img = cv2.resize(image, (new_w, self.model.img_h))

        normalized_img = resized_img.astype(np.float32) / 255.0
        tensor_img = torch.from_numpy(normalized_img)
        tensor_img = tensor_img.unsqueeze(0).unsqueeze(0).to(self.device)
        return tensor_img

    def decode_ctc(self, output: torch.Tensor) -> str:
        _, max_indices = torch.max(output, dim=2)
        indices = max_indices.squeeze().cpu().numpy()

        decoded = []
        prev_idx = -1
        for idx in indices:
            if idx == 0:
                continue
            if idx != prev_idx:
                decoded.append(self.charset[idx])
            prev_idx = idx
        return ''.join(decoded)

    def predict(self, image_path: str) -> str:

        input_tensor = self.preprocess(image_path)
        with torch.no_grad():
            output = self.model(input_tensor)
        return self.decode_ctc(output)


if __name__ == "__main__":
    CHECKPOINT = "../outputs/checkpoints/latest.pth"
    CHARSET_PATH = "../data/charset.txt"
    IMG_H = 32

    recognizer = CRNNInference(
        checkpoint_path=CHECKPOINT,
        charset_path=CHARSET_PATH,
        img_h=IMG_H,
        device="xpu"
    )

    test_images = [
        "../data/synthetic/test000.png",
        "../data/synthetic/test099.png"
    ]

    print("字符集内容：", recognizer.charset)
    print("字符索引映射示例：")
    print("'t'的索引:", recognizer.char_to_idx['t'])
    print("'e'的索引:", recognizer.char_to_idx['e'])
    print("'0'的索引:", recognizer.char_to_idx['0'])

    for img_path in test_images:
        if Path(img_path).exists():
            pred = recognizer.predict(img_path)
            print(f"图像：{Path(img_path).name} → 预测：{pred}")
        else:
            print(f"文件不存在：{img_path}")