# inference.py
from PIL import Image
import torch


def generate_and_display(pipe, prompt):
    with torch.no_grad():
        result = pipe(prompt)
        image = result.images[0].convert("RGB")

    return image


def generate_image(pipe, prompt="Naruto Uzumaki with a glowing aura", steps=50, scale=7.5):
    with torch.no_grad():
        result = pipe(prompt=prompt, num_inference_steps=steps, guidance_scale=scale)
        return result.images[0]
