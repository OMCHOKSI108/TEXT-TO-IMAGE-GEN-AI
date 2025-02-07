import streamlit as st
import random
import sys
from diffusers import DiffusionPipeline
from io import BytesIO
import os
os.environ["STREAMLIT_WATCH_FILE"] = "false"

# Function to generate images
def generate_image(prompt, use_refiner):
    import torch  # Import inside the function
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16" if device == "cuda" else None,
    ).to(device)

    if use_refiner:
        refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=pipe.text_encoder_2,
            vae=pipe.vae,
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16" if device == "cuda" else None,
        ).to(device)
        pipe.enable_model_cpu_offload()

    seed = random.randint(0, sys.maxsize)
    images = pipe(
        prompt=prompt,
        output_type="latent" if use_refiner else "pil",
        generator=torch.Generator(device).manual_seed(seed),
    ).images

    if use_refiner:
        images = refiner(prompt=prompt, image=images).images

    return images[0], prompt, seed

# Streamlit UI
st.title('Text-to-Image Generator')

prompt = st.text_input("Enter your prompt:")
use_refiner = st.checkbox('Use Refiner')

if st.button('Generate'):
    try:
        image, prompt, seed = generate_image(prompt, use_refiner)
        st.image(image, caption=f'Prompt: {prompt}\nSeed: {seed}', use_column_width=True)
        image.save("output.jpg")
        st.success('Image generated and saved as output.jpg')

        # Create a BytesIO object for image download
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
    except Exception as e:
        st.error(f"An error occurred: {e}")
