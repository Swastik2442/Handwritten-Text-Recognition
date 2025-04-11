import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from .config import IDX2CHAR, IMAGE_SIZE

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor()
])

class CRNN(nn.Module):
    def __init__(self, num_of_characters: int):
        super(CRNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.3)
        )
        self.fc1 = nn.Linear(128 * 8, 64)
        self.lstm1 = nn.LSTM(64, 256, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        self.fc2 = nn.Linear(512, num_of_characters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        b, c, h, w = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(b, h, c * w)

        x = self.fc1(x)
        x = F.relu(x)

        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        x = self.fc2(x)
        x = F.log_softmax(x, dim=2)
        return x

def ctc_greedy_decoder(logits: torch.Tensor):
    argmax_preds = torch.argmax(logits, dim=2)
    decoded = []
    for seq in argmax_preds:
        prev = -1
        chars = []
        for idx in seq:
            if idx.item() != prev and idx.item() != 0:
                chars.append(IDX2CHAR.get(idx.item(), ''))
            prev = idx.item()
        decoded.append(''.join(chars))
    return decoded
