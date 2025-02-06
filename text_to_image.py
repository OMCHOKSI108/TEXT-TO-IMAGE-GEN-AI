import torch
from diffusers import DiffusionPipeline
import random
import sys
import mediapy as media

def load_model():
    """Load the Stable Diffusion XL model with device fallback."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        use_safetensors=True,
        variant="fp16" if torch.cuda.is_available() else None,
    )
    return pipe.to(device)

def generate_image(prompt: str, seed: int = None):
    """Generate an image from a text prompt."""
    if seed is None:
        seed = random.randint(0, sys.maxsize)
    
    pipe = load_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device).manual_seed(seed)
    
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
