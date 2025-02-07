# Import necessary libraries
import streamlit as st
import mediapy as media
import random
import sys
import torch
from diffusers import DiffusionPipeline
from io import BytesIO
import platform

# Check the operating system
if platform.system() == 'Linux':
    st.warning('On Linux, make sure to install libgl1-mesa-glx: sudo apt-get install libgl1-mesa-glx')
elif platform.system() == 'Windows':
    st.warning('On Windows, ensure you have the OpenGL libraries installed.')

# Define a function to generate images
def generate_image(prompt, use_refiner):
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )

    if use_refiner:
        refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=pipe.text_encoder_2,
            vae=pipe.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )

        refiner = refiner.to("cuda")
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to("cuda")

    seed = random.randint(0, sys.maxsize)
    images = pipe(
        prompt=prompt,
        output_type="latent" if use_refiner else "pil",
        generator=torch.Generator("cuda").manual_seed(seed),
    ).images

    if use_refiner:
        images = refiner(prompt=prompt, image=images).images

    return images[0], prompt, seed

# Define the Streamlit app
st.title('Text-to-Image Generator')

prompt = st.text_input("Enter your prompt: ")
use_refiner = st.checkbox('Use Refiner')

if st.button('Generate'):
    image, prompt, seed = generate_image(prompt, use_refiner)
    st.image(image, caption=f'Prompt: {prompt}\nSeed: {seed}', use_column_width=True)
    image.save("output.jpg")
    st.success('Image generated and saved as output.jpg')

    # Create a BytesIO object and save the image in PNG format
    img_bytes = BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    # Provide a download button
    st.download_button(
        label="Download Image",
        data=img_bytes,
        file_name="generated_image.png",
        mime="image/png"
    )
