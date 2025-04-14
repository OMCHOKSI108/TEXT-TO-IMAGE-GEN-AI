# Text to Image Generation

![Pipeline](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/pipeline.png)

This project uses AI to generate images from text prompts. It requires a high-level GPU, which can be provided by Google Colab.

## How to Use

### Running on Google Colab

1. Open the [Google Colab](https://colab.research.google.com/) website.
2. Click on the **GitHub** tab.
3. Enter the URL of this repository: `https://github.com/OMCHOKSI108/TEXT-TO-IMAGE-GEN-AI`
4. Select the `text-to-image.ipynb` file.
5. Click on **Open Notebook**.

### Colab Setup

1. Make sure you have the correct runtime settings:
   - Go to **Runtime** > **Change runtime type**.
   - Select **GPU** as the hardware accelerator.
2. Run the cells in the notebook sequentially.

## Requirements

- Google Colab (for running the notebook with GPU)
- List of required libraries (automatically installed in Colab)

## Basic Code Explanation

- Loads Stable Diffusion XL model using Hugging Face's `diffusers` library.
- Takes a text prompt as input from the user.
- Generates an image based on the prompt using the GPU.
- Optionally refines the image for more detail (if enabled).
- Displays and saves the generated image (`output.jpg`).
