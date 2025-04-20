from tempfile import NamedTemporaryFile
from PIL import Image

import torch
import streamlit as st
from craft.create import TextRegions

from src.model import CRNN # pylint: disable=unused-import
from test_model import test_full

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load("weights/crnn_model_000.pkl", map_location=device)
craft = TextRegions(trained_model="weights/craft_mlt_25k.pth", show_time=False)

st.set_page_config("HTR App")
st.title("🖼️ Handwritten Text Recognition")

uploaded_file = st.file_uploader("Upload an Image", type=("png", "jpg", "jpeg"))

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Processing..."):
        file = NamedTemporaryFile("wb", suffix=".png", delete=False)
        image.save(file, "png")
        extracted_text = test_full(model, craft, file.name)[0]
        file.close()
    st.subheader("📄 Extracted Text:")
    st.code(extracted_text, language="text")
