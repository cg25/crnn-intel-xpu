import torch
import cv2
import numpy as np
from pathlib import Path
from torch.nn import CTCLoss
from core.crnn import CRNN


class CRNNValidator:
    def __init__(self, checkpoint_path: str, charset_path: str, img_h: int = 32, device: str = "xpu"):
        self.device = torch.device(device)
        assert torch.xpu.is_available(), "XPU not available"

        with open(charset_path, 'r', encoding='utf-8-sig') as f:
            self.charset = f.read().splitlines()

        self.model = CRNN(img_h=img_h, num_classes=len(self.charset)).to(self.device)
        self._load_checkpoint(checkpoint_path)
        self.model.eval()

        self.criterion = CTCLoss(blank=0)

        self.char_to_idx = {char: idx for idx, char in enumerate(self.charset)}

    def _load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        print(f"loaded checkpoint：{Path(path).name}")

    def _preprocess(self, image_path: str) -> torch.Tensor:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        h, w = image.shape
        new_w = int(w * (self.model.img_h / h))
        resized_img = cv2.resize(image, (new_w, self.model.img_h))

        normalized = resized_img.astype(np.float32) / 255.0
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).to(self.device)
        return tensor

    def _get_target(self, image_path: Path) -> torch.Tensor:
        label = image_path.stem
        indices = []
        for c in label:
            if c not in self.char_to_idx:
                raise ValueError(f"illegal character '{c}' at {image_path.name}")
            indices.append(self.char_to_idx[c])
        return torch.tensor(indices, dtype=torch.long)

    def calculate_validation_loss(self, validation_dir: str) -> float:
        self.model.eval()
        total_loss = 0.0
        count = 0

        image_paths = list(Path(validation_dir).glob("*.[pj][np]g"))

        with torch.no_grad():
            for img_path in image_paths:
                try:
                    input_tensor = self._preprocess(str(img_path))
                    target = self._get_target(img_path).to(self.device)

                    outputs = self.model(input_tensor)  # [seq_len, 1, num_classes]

                    input_length = torch.tensor([outputs.size(0)], dtype=torch.long)
                    target_length = torch.tensor([target.size(0)], dtype=torch.long)

                    loss = self.criterion(
                        outputs.log_softmax(2),  # [T, N, C]
                        target.unsqueeze(0),  # [N, S]
                        input_length,
                        target_length
                    )

                    total_loss += loss.item()
                    count += 1

                except Exception as e:
                    print(f"handling {img_path.name} got error：{str(e)}")

        return total_loss / count if count > 0 else float('nan')


if __name__ == "__main__":
    CHECKPOINT = "../outputs/checkpoints/latest.pth"
    CHARSET_PATH = "../data/charset.txt"
    VAL_DIR = "../data/validation"
    IMG_H = 32

    validator = CRNNValidator(
        checkpoint_path=CHECKPOINT,
        charset_path=CHARSET_PATH,
        img_h=IMG_H,
        device="xpu"
    )

    val_loss = validator.calculate_validation_loss(VAL_DIR)
    print(f"average validation loss：{val_loss:.4f}")