# data_utils.py
from datasets import load_dataset, Dataset
from torchvision import transforms
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def load_and_preprocess_dataset(dataset_name="lambdalabs/naruto-blip-captions"):
    dataset = load_dataset(dataset_name, split="train")
    processed = []

    for example in tqdm(dataset, desc="Prétraitement du dataset"):
        image = example["image"].convert("RGB")
        image_tensor = transform(image)
        processed.append({"pixel_values": image_tensor, "text": example["text"]})

    return Dataset.from_list(processed)

def collate_fn(batch, tokenizer):
    images = torch.stack([
        torch.tensor(example["pixel_values"]) if not isinstance(example["pixel_values"], torch.Tensor) else example["pixel_values"]
        for example in batch
    ])
    prompts = [example["text"] for example in batch]

    inputs = tokenizer(prompts, padding="max_length", truncation=True, max_length=64, return_tensors="pt")

    return {
        "pixel_values": images,
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
    }

def check_batch(dataset, tokenizer):
    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer))
    for batch in dataloader:
        print(f"✅ Image tensor shape: {batch['pixel_values'].shape}")
        print(f"✅ Input IDs shape: {batch['input_ids'].shape}")

        img = batch["pixel_values"][0].permute(1, 2, 0)
        img = (img * 0.5 + 0.5).clamp(0, 1)

        plt.imshow(img.cpu())
        plt.title("Aperçu image après transformation")
        plt.axis("off")
        plt.show()
        break