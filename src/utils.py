from typing import Callable, Sequence
from PIL import Image

import torch

from .config import CHAR2IDX, IDX2CHAR, IMAGE_SIZE # type: ignore

TransformFnType = Callable[[Image.Image], Image.Image]

def encode_text(text: str):
    return [CHAR2IDX.get(char, CHAR2IDX['<unk>']) for char in text]

def decode_text(labels: Sequence[int]):
    if isinstance(labels, Sequence):
        return [IDX2CHAR.get(idx, '<unk>') for idx in labels]
    return [IDX2CHAR.get(labels, '<unk>')]

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

def to_time_string(ts: int):
    if ts < 1e+3:
        return f"{ts}ns"
    if ts < 1e+6:
        return f"{ts // 1e+3:.0f}us"
    if ts < 1e+9:
        return f"{ts // 1e+6:.0f}ms"
    if ts < 6e+10:
        return f"{ts // 1e+9:.0f}s"
    if ts < 36e+11:
        tsf = round(ts / 6e+10, 1)
        tsf = int(tsf) if tsf == int(tsf) else tsf
        return f"{tsf}min{'s' if tsf != 1 else ''}"
    tsf = round(ts / 36e+11, 1)
    tsf = int(tsf) if tsf == int(tsf) else tsf
    return f"{tsf}hr{'s' if tsf != 1 else ''}"
