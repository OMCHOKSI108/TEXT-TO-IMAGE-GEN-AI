import streamlit as st
import torch
from diffusers import DiffusionPipeline
import random
from PIL import Image

# Load the model
@st.cache_resource
def load_model():
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    return pipe.to("cuda")

pipe = load_model()

st.title("Text-to-Image AI")

prompt = st.text_input("Enter your prompt:", "A beautiful futuristic city at sunset")

if st.button("Generate Image"):
    seed = random.randint(0, 999999)
    generator = torch.Generator("cuda").manual_seed(seed)

    with st.spinner("Generating..."):
        image = pipe(prompt, generator=generator).images[0]
    
    st.image(image, caption=f"Generated Image (Seed: {seed})", use_column_width=True)
    image.save("output.jpg")
    st.success("Image generated successfully!")
