import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, img_h: int, num_classes: int):
        super().__init__()
        self.img_h = img_h

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [64, h/2, w/2]
            nn.Conv2d(64, 128, 3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [128, h/4, w/4]
            nn.Conv2d(128, 256, 3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d((2, 1))  # [256, h/8, w/4]
        )

        self.rnn = nn.LSTM(
            input_size=256 * (img_h // 8),
            hidden_size=256,
            bidirectional=True,
            num_layers=2
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)  # [b,256,h/8,w/4]
        x = x.permute(3, 0, 1, 2)  # [w/4,b,256,h/8]
        x = x.view(x.size(0), x.size(1), -1)  # [w/4,b,256*(h/8)]

        x, _ = self.rnn(x)  # [seq_len,b,512]
        x = self.fc(x)  # [seq_len,b,num_classes]
        return torch.log_softmax(x, dim=2)