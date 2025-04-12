# # metrics_utils.py
# import torch
# from torchvision.transforms.functional import resize
# from piq import FID, LPIPS
# from torchmetrics.image.fid import FrechetInceptionDistance
# from tqdm import tqdm
# import torchvision.transforms as transforms

# @torch.no_grad()
# def generate_comparison_images(teacher, student, prompts, device):
#     teacher_images, student_images = [], []
#     to_tensor = transforms.ToTensor()

#     for prompt in tqdm(prompts, desc="Génération d'images pour métriques"):
#         teacher_result = teacher(prompt=prompt, num_inference_steps=50, guidance_scale=7.5)
#         student_result = student(prompt=prompt, num_inference_steps=25, guidance_scale=7.5)

#         img_teacher = to_tensor(teacher_result.images[0]).unsqueeze(0).to(device)
#         img_student = to_tensor(student_result.images[0]).unsqueeze(0).to(device)

#         teacher_images.append(img_teacher)
#         student_images.append(img_student)

#     teacher_images = torch.cat(teacher_images, dim=0)
#     student_images = torch.cat(student_images, dim=0)
#     return teacher_images, student_images

# @torch.no_grad()
# def compute_metrics(teacher_images, student_images, device="cuda"):
#     teacher_images = resize(teacher_images, [299, 299])
#     student_images = resize(student_images, [299, 299])

#     fid = FrechetInceptionDistance(feature=2048).to(device)
#     fid.update((student_images * 255).byte(), real=False)
#     fid.update((teacher_images * 255).byte(), real=True)
#     fid_score = fid.compute().item()

#     lpips = LPIPS().to(device)
#     lpips_score = lpips(student_images, teacher_images).mean().item()

#     print("\n📊 Comparaison Student vs Teacher:")
#     print(f"FID Score : {fid_score:.4f}")
#     print(f"LPIPS Score : {lpips_score:.4f} (plus proche de 0 = plus proche visuellement)")

#     return fid_score, lpips_score


# metrics_utils.py
import torch
from torchvision.transforms.functional import resize
from piq import LPIPS
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

@torch.no_grad()
def generate_comparison_images(teacher, student, prompts, device):
    teacher_images, student_images = [], []
    image_pairs = []
    to_tensor = transforms.ToTensor()

    for prompt in tqdm(prompts, desc="Génération d'images pour métriques"):
        teacher_result = teacher(prompt=prompt, num_inference_steps=50, guidance_scale=7.5)
        student_result = student(prompt=prompt, num_inference_steps=25, guidance_scale=7.5)

        img_teacher = to_tensor(teacher_result.images[0]).unsqueeze(0).to(device)
        img_student = to_tensor(student_result.images[0]).unsqueeze(0).to(device)

        teacher_images.append(img_teacher)
        student_images.append(img_student)
        image_pairs.append((prompt, teacher_result.images[0], student_result.images[0]))

    teacher_images = torch.cat(teacher_images, dim=0)
    student_images = torch.cat(student_images, dim=0)
    return teacher_images, student_images, image_pairs

@torch.no_grad()
def compute_metrics(teacher_images, student_images, image_pairs, device="cuda", top_k=3):
    teacher_images = resize(teacher_images, [299, 299])
    student_images = resize(student_images, [299, 299])

    fid = FrechetInceptionDistance(feature=2048).to(device)
    fid.update((student_images * 255).byte(), real=False)
    fid.update((teacher_images * 255).byte(), real=True)
    fid_score = fid.compute().item()

    lpips = LPIPS().to(device)
    lpips_scores = lpips(student_images, teacher_images).detach().cpu().view(-1)

    print("\n📊 Comparaison Student vs Teacher:")
    print(f"FID Score : {fid_score:.4f}")
    print(f"LPIPS Score (moyen) : {lpips_scores.mean().item():.4f} (plus proche de 0 = plus proche visuellement)")

    # Extraire les meilleurs paires (student le plus proche du teacher)
    best_indices = torch.topk(-lpips_scores, top_k).indices  # - pour les plus faibles valeurs

    for idx in best_indices:
        prompt, teacher_img, student_img = image_pairs[idx]
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(teacher_img)
        axes[0].set_title(f"Teacher\nPrompt: {prompt}")
        axes[0].axis("off")

        axes[1].imshow(student_img)
        axes[1].set_title(f"Student\nLPIPS: {lpips_scores[idx]:.4f}")
        axes[1].axis("off")

        plt.suptitle("🧠 Meilleure correspondance visuelle")
        plt.tight_layout()
        plt.show()

    return fid_score, lpips_scores.mean().item()
