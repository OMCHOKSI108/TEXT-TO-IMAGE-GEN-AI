# Text-to-Image AI

This project is a **Text-to-Image AI** using Stable Diffusion XL, deployed with **Streamlit**. It allows users to generate high-quality images from text prompts.

## Features
- Generates images from text prompts using Stable Diffusion XL.
- Deployed using **Streamlit**.
- GPU-accelerated inference for faster image generation.
- Easy to use interface.

## File Structure
```
text-to-image-ai/
│── app.py                   # Main Streamlit app
│── model.py                 # Model loading and inference
│── requirements.txt         # Dependencies
│── README.md                # Project documentation
│── .gitignore               # Ignore unnecessary files
│── images/                  # Store generated images (optional)
│── notebooks/
│   └── text-to-image.ipynb  # Original Google Colab notebook
```

## Setup & Installation
### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/text-to-image-ai.git
cd text-to-image-ai
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App
```bash
streamlit run app.py
```

## Deploying on Streamlit Cloud
1. Push your repository to GitHub.
2. Go to **[Streamlit Cloud](https://share.streamlit.io/)** and log in.
3. Click **"New app"**, connect your GitHub repo, and select `app.py` as the entry point.
4. Click **"Deploy"**.

## Deploying on Hugging Face Spaces
1. Create a new space on **[Hugging Face Spaces](https://huggingface.co/spaces)**.
2. Select **Streamlit** as the framework.
3. Upload your code and `requirements.txt`.
4. Click **Deploy**.

## Example Prompt
```
"A futuristic cyberpunk city with neon lights and flying cars."
```

## Output Example
![Example Image](images/example.jpg)

## License
This project is licensed under the MIT License.

---

Made with ❤️ by [Your Name]

