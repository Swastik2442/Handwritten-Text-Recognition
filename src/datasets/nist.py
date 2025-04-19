import os

from torch.utils.data import Dataset

from ..config import HEX2CHAR # type: ignore
from .common import get_item, TransformFnType

class NISTDataset(Dataset):
    """A PyTorch Dataset for loading NIST Special Database 19"""

    def __init__(self, images_dir: str, transform: TransformFnType | None = None, **_):
        self.transform = transform
        self.samples: list[tuple[str, str]] = []

        for class_id in os.listdir(images_dir):
            class_path = os.path.join(images_dir, class_id)
            if not os.path.isdir(class_path):
                continue
            char = HEX2CHAR[class_id.lower()]

            for subfolder in os.listdir(class_path):
                subfolder_path = os.path.join(class_path, subfolder)
                if not os.path.isdir(subfolder_path):
                    continue

                for file in os.listdir(subfolder_path):
                    if file.lower().endswith((".png", ".jpg", ".jpeg")):
                        img_path = os.path.join(subfolder_path, file)
                        self.samples.append((img_path, char))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return get_item(*self.samples[idx], self.transform)
