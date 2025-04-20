import random
from PIL import Image

import numpy as np
import cv2
import torchvision.transforms as torchtransforms

from ..config import IMAGE_SIZE # type: ignore

class ElasticTransform:
    def __init__(self, alpha: float, sigma: float):
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, img):
        image = np.array(img)

        random_state = np.random.RandomState(None)
        shape = image.shape

        dx = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1), (0, 0), self.sigma) * self.alpha
        dy = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1), (0, 0), self.sigma) * self.alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)

        distorted = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return Image.fromarray(distorted)

class RandomCutout:
    def __init__(self, size=20, p=0.5):
        self.size = size
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img

        img = np.array(img)
        h, w = img.shape[:2]

        x = random.randint(0, w - self.size)
        y = random.randint(0, h - self.size)

        img[y:y+self.size, x:x+self.size] = 255  # white cutout
        return Image.fromarray(img)

aug_transforms = torchtransforms.Compose([
    torchtransforms.Resize(IMAGE_SIZE),
    ElasticTransform(alpha=34, sigma=4),
    RandomCutout(size=30),
    torchtransforms.ToTensor(),
    torchtransforms.Normalize((0.5,), (0.5,))
])
