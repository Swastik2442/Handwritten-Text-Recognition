import os
from PIL import Image, UnidentifiedImageError

from torch.utils.data import Dataset

from transforms import aug_transforms
from ..utils import get_item, TransformFnType # type: ignore

class IAMDataset(Dataset):
    """A PyTorch Dataset for loading IAM Handwriting Dataset for Lines/Sentences/Words"""

    def __init__(self, labels_path: str, images_dir: str, base_transform: TransformFnType | None = None, augment: bool = True, **_):
        self.images_dir = images_dir
        self.base_transform = base_transform
        self.augment = augment
        self.samples = self.parse_lines_txt(labels_path)

    def parse_lines_txt(self, txt_path: str):
        samples: list[tuple[str, str]] = []
        with open(txt_path, 'r', encoding='utf8') as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) < 9 or parts[1] != 'ok':
                    continue

                img_path = self._get_img_path_(parts)
                try:
                    Image.open(img_path)
                except UnidentifiedImageError:
                    print("Error opening ", img_path, "- removing")
                    os.remove(img_path)
                    continue
                except FileNotFoundError:
                    print("No such Image", img_path)
                    continue

                label = self._get_label_(parts)
                samples.append((img_path, label))
        return samples

    def _get_img_path_(self, parts: list[str]):
        img_id = parts[0]
        img_id_sep = img_id.split("-")
        dir_loc = os.path.join(img_id_sep[0], f"{img_id_sep[0]}-{img_id_sep[1]}")
        return os.path.join(self.images_dir, dir_loc, f"{img_id}.png")

    def _get_label_(self, parts: list[str]):
        raise NotImplementedError()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return get_item(*self.samples[idx], aug_transforms if self.augment else self.base_transform)

class IAMWordsDataset(IAMDataset):
    """A PyTorch Dataset for loading IAM Handwriting Dataset for Words"""

    def _get_label_(self, parts: list[str]):
        return ' '.join(parts[8:])

class IAMLinesDataset(IAMDataset):
    """A PyTorch Dataset for loading IAM Handwriting Dataset for Lines/Sentences"""

    def _get_label_(self, parts: list[str]):
        return ' '.join(parts[8].split("|"))
