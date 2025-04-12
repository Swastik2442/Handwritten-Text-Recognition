import os
from typing import Callable
from PIL import Image, UnidentifiedImageError

import torch
from torch.utils.data import Dataset

from ..config import CHAR2IDX, IMAGE_SIZE # type: ignore

TransformFnType = Callable[[Image.Image], Image.Image]

class IAMDataset(Dataset):
    """A PyTorch Dataset for loading IAM Handwriting Dataset for Words"""

    def __init__(self, labels_path: str, images_dir: str, transform: TransformFnType | None = None):
        self.images_dir = images_dir
        self.transform = transform
        self.samples = self.parse_lines_txt(labels_path)

    def parse_lines_txt(self, txt_path: str):
        samples = []
        with open(txt_path, 'r') as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) < 9 or parts[1] != 'ok':
                    continue

                img_id = parts[0]
                img_id_sep = img_id.split("-")
                dir_loc = os.path.join(img_id_sep[0], f"{img_id_sep[0]}-{img_id_sep[1]}")
                img_path = os.path.join(self.images_dir, dir_loc, f"{img_id}.png")

                try:
                    Image.open(img_path)
                except UnidentifiedImageError:
                    print("Error opening ", img_path, "- removing")
                    os.remove(img_path)
                    continue
                except FileNotFoundError:
                    print("No such Image", img_path)
                    continue

                label = ' '.join(parts[8:])
                samples.append((img_path, label))
        return samples

    def encode_text(self, text: str):
        return [CHAR2IDX.get(char, CHAR2IDX['<unk>']) for char in text]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, text = self.samples[idx]
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)

        label = self.encode_text(text)
        label_length = len(label)
        input_length = IMAGE_SIZE[1]

        return image, torch.tensor(label), torch.tensor(input_length), torch.tensor(label_length)

def collate_fn(batch):
    images, labels, input_lengths, label_lengths = zip(*batch)
    images_tensor = torch.stack(images)
    input_lengths_tensor = torch.stack(input_lengths)
    label_lengths_tensor = torch.stack(label_lengths)
    flat_labels_tensor = torch.cat(labels)
    return images_tensor, flat_labels_tensor, input_lengths_tensor, label_lengths_tensor
