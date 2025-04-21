import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from .config import IDX2CHAR, IMAGE_SIZE

base_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor()
])

class CRNN(nn.Module):
    """A Convolutional Recurrent Neural Network Model for Handwritten Text Recognition"""

    def __init__(self, num_of_characters: int): # 2,444,800 + num_of_characters
        super(CRNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), # 288 + 32 = 320
            nn.BatchNorm2d(32),                         # 32 + 32 = 64
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 18,432 + 64 = 18,496
            nn.BatchNorm2d(64),                          # 64 + 64 = 128
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 73,728 + 128 = 73,856
            nn.BatchNorm2d(128),                          # 128 + 128 = 256
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.3)
        )
        self.fc1 = nn.Linear(128 * 8, 64)                                    # 65,536 + 64 = 65,600
        self.lstm1 = nn.LSTM(64, 256, bidirectional=True, batch_first=True)  # 2 x (65,536 + 262,144 + 1024 + 1024) = 2 x 329,727 = 659,454
        self.lstm2 = nn.LSTM(512, 256, bidirectional=True, batch_first=True) # 2 x (524,288 + 262,144 + 1024 + 1024) = 2 x 788,480 = 1,576,960
        self.fc2 = nn.Linear(512, num_of_characters)                         # 49,664 + num_of_characters

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
    decoded: list[str] = []
    for seq in argmax_preds:
        prev = -1
        chars = []
        for idx in seq:
            if idx.item() != prev and idx.item() != 0:
                chars.append(IDX2CHAR.get(idx.item(), "<unk>"))
            prev = idx.item()
        decoded.append(''.join(chars))
    return decoded
