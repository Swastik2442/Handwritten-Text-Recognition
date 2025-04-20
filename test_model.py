from PIL import Image

import fastwer
import torch
import textblob

from craft.create import TextRegions
from refine import TextRefiner
from src.model import CRNN, base_transform, ctc_greedy_decoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_num_params(model: torch.nn.Module, show_params=True):
    tot_params = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        tot_params += num_params
        if show_params:
            print(f"{name:^50} {num_params:^10}")
    return tot_params

def get_scores(actual: str, pred: str):
    wer = fastwer.score_sent(actual, pred)
    cer = fastwer.score_sent(actual, pred, char_level=True)
    return wer, cer

def test_model(model: CRNN, img_path: str):
    image = Image.open(img_path).convert('L')
    image_tensor = base_transform(image).unsqueeze(0).to(device)

    model.eval()
    output = model(image_tensor)
    return ctc_greedy_decoder(output.cpu())[0]

def _polys_to_image(orig_img: Image.Image, polys):
    img = orig_img.copy()
    (x1, y1), _, (x3, y3), _ = polys
    x1, y1, x3, y3 = tuple(map(round, (x1, y1, x3, y3)))
    return img.crop((x1, y1, x3, y3))

def test_full(crnn: CRNN, craft: TextRegions, img_path: str):
    polys = craft.detectRegions(img_path)
    polys = tuple(sorted(polys, key=lambda x: x[0][0]))

    img = Image.open(img_path).convert('L')
    imgs = (_polys_to_image(img, poly) for poly in polys)
    imgs = (base_transform(img).unsqueeze(0).to(device) for img in imgs)

    crnn.eval()
    text = ' '.join([ctc_greedy_decoder(crnn(img).cpu())[0] for img in imgs])
    sc_text = textblob.TextBlob(text).correct().string
    ref_text = TextRefiner().refine_text(sc_text)

    return text, sc_text, ref_text

if __name__ == '__main__':
    model = torch.load("weights/crnn_model_000.pkl", map_location=device, weights_only=False)
    craft = TextRegions(trained_model="weights/craft_mlt_25k.pth", show_time=False)

    ans = test_full(model, craft, "data/a03-006-00.png")
    label = ' '.join("Today|,|for|example|,|the|Foreign|Minister|of|Indo-".split("|"))
    print(label, *ans, get_scores(label, ans[0]), get_scores(label, ans[1]), sep='\n')
