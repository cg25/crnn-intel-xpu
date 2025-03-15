from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class CRNNDataset(Dataset):
    def __init__(self, data_root: str, img_h: int = 32, charset_path: str = "data/charset.txt"):
        self.img_paths = sorted(list(Path(data_root).glob("*.png")))
        assert len(self.img_paths) > 0, f"No PNG images found in {data_root}"

        with open(charset_path, 'r', encoding='utf-8') as f:
            self.charset = f.read().splitlines()
        self.char_to_idx = {char: idx for idx, char in enumerate(self.charset)}

        self._validate_labels()

        self.img_h = img_h

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]

        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

        h, w = image.shape
        new_w = int(w * (self.img_h / h))
        resized_img = cv2.resize(image, (new_w, self.img_h))

        normalized_img = resized_img.astype(np.float32) / 255.0
        tensor_img = torch.from_numpy(normalized_img).unsqueeze(0)

        label = path.stem

        target = [self.char_to_idx[c] for c in label]

        return tensor_img, torch.tensor(target)

    def _validate_labels(self):
        invalid_samples = []

        for path in self.img_paths:
            label = path.stem
            for c in label:
                if c not in self.char_to_idx:
                    invalid_samples.append((
                        str(path),
                        f"字符 '{c}' 不在字符集中（允许字符：{self.charset}）"
                    ))

        if invalid_samples:
            error_msg = "\n".join([f"{path}: {reason}" for path, reason in invalid_samples])
            raise ValueError(f"发现无效标签字符：\n{error_msg}")

    @property
    def num_classes(self):
        return len(self.charset)


if __name__ == "__main__":
    dataset = CRNNDataset(data_root="data/synthetic", img_h=32)
    print(f"数据集大小：{len(dataset)}")

    img, target = dataset[0]
    print(f"图像张量形状：{img.shape}")
    print(f"标签索引：{target.tolist()}")
    print(f"原始标签：{dataset.img_paths[0].stem}")