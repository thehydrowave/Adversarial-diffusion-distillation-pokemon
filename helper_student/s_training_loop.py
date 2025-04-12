# training_loop.py
import torch
from tqdm import tqdm
import torch.nn as nn

@torch.no_grad()
def generate_latents_and_images(student, scheduler, x_t, t, encoder_hidden_states):
    noise_pred = student.unet(x_t, t, encoder_hidden_states).sample
    latents_fake = scheduler.step(noise_pred.cpu(), t.cpu(), x_t.cpu()).prev_sample.to(x_t.device)
    gen_images = student.vae.decode(latents_fake.half()).sample
    return (gen_images / 2 + 0.5).clamp(0, 1)

def train_student_with_adversarial(
    student, teacher, dataloader, discriminator, scheduler,
    optimizer_student, optimizer_disc,
    lambda_adv=0.5, device="cuda", epochs=2
):
    adversarial_loss_fn = nn.BCEWithLogitsLoss()
    mse_loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch}")
        for pixel_values, prompts in loop:
            pixel_values = pixel_values.to(device, dtype=torch.float16)
            batch_size = pixel_values.shape[0]

            # Texte -> embedding
            inputs = teacher.tokenizer(prompts, padding="max_length", truncation=True, return_tensors="pt").to(device)
            encoder_hidden_states = teacher.text_encoder(**inputs).last_hidden_state

            # Image -> latent + bruit
            t = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,), device=device).long()
            latents = teacher.vae.encode(pixel_values).latent_dist.sample() * 0.18215
            noise = torch.randn_like(latents)
            x_t = scheduler.add_noise(latents, noise, t)
            target = noise

            # === Génération d'image ===
            student.unet.eval()
            gen_images = generate_latents_and_images(student, scheduler, x_t, t, encoder_hidden_states)

            # === Discriminateur ===
            discriminator.train()
            discriminator.zero_grad()

            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            real_output = discriminator(pixel_values.float())
            fake_output = discriminator(gen_images.detach().float())

            d_loss_real = adversarial_loss_fn(real_output, real_labels)
            d_loss_fake = adversarial_loss_fn(fake_output, fake_labels)
            d_loss = (d_loss_real + d_loss_fake) / 2

            d_loss.backward()
            optimizer_disc.step()

            # === Student ===
            student.unet.train()
            optimizer_student.zero_grad()

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                noise_pred = student.unet(x_t, t, encoder_hidden_states).sample
                loss_sds = mse_loss_fn(noise_pred, target)

                gen_output = discriminator(gen_images.float())
                loss_adv = adversarial_loss_fn(gen_output, real_labels)

                s_loss = loss_sds + lambda_adv * loss_adv

            s_loss.backward()
            optimizer_student.step()

            loop.set_postfix(d_loss=d_loss.item(), s_loss=s_loss.item(), sds=loss_sds.item())
