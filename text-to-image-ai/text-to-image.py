import torch
from diffusers import DiffusionPipeline
import random
import sys
import mediapy as media

def load_model():
    """Load the Stable Diffusion XL model."""
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    return pipe.to("cuda")

def generate_image(prompt: str, seed: int = None):
    """Generate an image from a text prompt."""
    if seed is None:
        seed = random.randint(0, sys.maxsize)
    
    pipe = load_model()
    generator = torch.Generator("cuda").manual_seed(seed)
    
    images = pipe(
        prompt=prompt,
        output_type="pil",
        generator=generator,
    ).images
    
    images[0].save("output.jpg")
    return images, seed

if __name__ == "__main__":
    prompt = input("Enter your prompt: ")
    images, seed = generate_image(prompt)
    print(f"Prompt: {prompt}\nSeed: {seed}")
    media.show_images(images)
