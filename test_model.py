from PIL import Image

import torch

from craft.create import TextRegions
from src.model import CRNN, transform, ctc_greedy_decoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_num_params(model: torch.nn.Module, show_params=True):
    tot_params = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        tot_params += num_params
        if show_params:
            print(f"{name:^50} {num_params:^10}")
    return tot_params

def test_model(model: CRNN, img_path: str):
    image = Image.open(img_path).convert('L')
    image_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    output = model(image_tensor)
    return ctc_greedy_decoder(output.cpu())[0]

def _polys_to_image(orig_img: Image.Image, polys):
    img = orig_img.copy()
    (x1, y1), _, (x3, y3), _ = polys
    x1, y1, x3, y3 = tuple(map(round, (x1, y1, x3, y3)))
    return img.crop((x1, y1, x3, y3))

def test_full(crnn: CRNN, craft: TextRegions, img_path: str):
    ans = craft.detectRegions(img_path)
    ans = tuple(sorted(ans, key=lambda x: x[0][0]))

    img = Image.open(img_path).convert('L')
    imgs = (_polys_to_image(img, poly) for poly in ans)
    imgs = (transform(img).unsqueeze(0).to(device) for img in imgs)

    crnn.eval()
    return ' '.join([ctc_greedy_decoder(crnn(img).cpu())[0] for img in imgs])

if __name__ == '__main__':
    model = torch.load("weights/crnn_model.pkl", map_location=device)
    craft = TextRegions(trained_model="weights/craft_mlt_25k.pth", show_time=False)
    print(test_full(model, craft, "data/a01-000u.png"))
