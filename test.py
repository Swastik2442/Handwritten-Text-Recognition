from PIL import Image

import torch

from src.model import CRNN, transform, ctc_greedy_decoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_model(model: CRNN, img_path: str):
    image = Image.open(img_path).convert('L')
    image_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    output = model(image_tensor)
    return ctc_greedy_decoder(output.cpu())[0]
