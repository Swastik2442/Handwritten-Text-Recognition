from typing import Callable, Sequence
from PIL import Image

import torch

from ..config import CHAR2IDX, IDX2CHAR, IMAGE_SIZE # type: ignore

TransformFnType = Callable[[Image.Image], Image.Image]

def encode_text(text: str):
    return [CHAR2IDX.get(char, CHAR2IDX['<unk>']) for char in text]

def decode_text(labels: Sequence[int]):
    if isinstance(labels, Sequence):
        return ' '.join(IDX2CHAR.get(char, '<unk>') for char in labels)
    return IDX2CHAR.get(labels, '<unk>')

def get_item(img_path: str, text: str, transform: TransformFnType | None = None):
    image = Image.open(img_path).convert("L")
    if transform:
        image = transform(image)

    label = encode_text(text)
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
