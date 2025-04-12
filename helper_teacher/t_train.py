
# # train.py
# import os
# import torch
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from t_data_utils import collate_fn
# from t_visual_utils import plot_loss_curve, smooth_curve

# def train(pipe, dataset, tokenizer, lora_params, output_dir, gradient_accumulation_steps=2, epochs=3, batch_size=1):
#     dataloader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         collate_fn=lambda b: collate_fn(b, tokenizer)
#     )

#     optimizer = torch.optim.Adam(lora_params, lr=1e-4)
#     loss_history = []

#     for epoch in range(epochs):
#         loop = tqdm(dataloader, desc=f"Epoch {epoch}")
#         for step, batch in enumerate(loop):
#             pixel_values = batch["pixel_values"].type(torch.float16).to("cuda")
#             input_ids = batch["input_ids"].to("cuda")
#             attention_mask = batch["attention_mask"].to("cuda")

#             noise = torch.randn_like(pixel_values)
#             noised_images = pixel_values + noise

#             timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (pixel_values.shape[0],)).long().to("cuda").type(torch.float16)
#             encoder_hidden_states = pipe.text_encoder(input_ids)[0].type(torch.float16)

#             noised_images = torch.cat([noised_images, torch.zeros_like(noised_images[:, :1, :, :])], dim=1)

#             model_output = pipe.unet(noised_images, timesteps, encoder_hidden_states).sample
#             # loss = model_output.mean()
            
#             noise = torch.randn_like(pixel_values)
#             noised_images = pixel_values + noise
#             target = noise  # c'est ce que le modèle doit prédire

#             # Prédiction du bruit par le modèle
#             model_output = pipe.unet(noised_images, timesteps, encoder_hidden_states).sample

            
#             mse_loss_fn = torch.nn.MSELoss()
#             loss = mse_loss_fn(model_output, target)

#             loss = loss / gradient_accumulation_steps
#             loss.backward()

#             if (step + 1) % gradient_accumulation_steps == 0:
#                 optimizer.step()
#                 optimizer.zero_grad()

#             loop.set_postfix(loss=loss.item())
#             loss_history.append(loss.item())

#     # Sauvegarde des poids LoRA
#     lora_state_dict = {}
#     for name, processor in pipe.unet.attn_processors.items():
#         if hasattr(processor, 'state_dict'):
#             for k, v in processor.state_dict().items():
#                 lora_state_dict[f"{name}.{k}"] = v

#     os.makedirs(output_dir, exist_ok=True)
#     torch.save(lora_state_dict, os.path.join(output_dir, "teacher_lora_weights_clean.bin"))
#     print("Poids LoRA sauvegardés dans :", os.path.join(output_dir, "teacher_lora_weights.bin"))

#     # Affichage de la courbe de convergence
#     smoothed = smooth_curve(loss_history)
#     plot_loss_curve(smoothed)

#     return loss_history


# train.py
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from t_data_utils import collate_fn
from t_visual_utils import plot_loss_curve, smooth_curve

def train(pipe, dataset, tokenizer, lora_params, output_dir, gradient_accumulation_steps=2, epochs=3, batch_size=1):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer)
    )

    optimizer = torch.optim.Adam(lora_params, lr=1e-4)
    mse_loss_fn = torch.nn.MSELoss()
    loss_history = []

    for epoch in range(epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch}")
        for step, batch in enumerate(loop):
            pixel_values = batch["pixel_values"].type(torch.float16).to("cuda")
            input_ids = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")

            # Génère du bruit et l'image bruitée
            noise = torch.randn_like(pixel_values)
            noised_images = pixel_values + noise
            target = noise  # ce que le modèle doit apprendre à prédire

            timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (pixel_values.shape[0],)).long().to("cuda").type(torch.float16)
            encoder_hidden_states = pipe.text_encoder(input_ids)[0].type(torch.float16)

            # Fusion des canaux (si nécessaire)
            noised_images = torch.cat([noised_images, torch.zeros_like(noised_images[:, :1, :, :])], dim=1)

            # Prédiction
            model_output = pipe.unet(noised_images, timesteps, encoder_hidden_states).sample
            loss = mse_loss_fn(model_output, target)

            # Backpropagation
            loss = loss / gradient_accumulation_steps
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            loop.set_postfix(loss=loss.item())
            loss_history.append(loss.item())

    # Sauvegarde des poids LoRA
    lora_state_dict = {}
    for name, processor in pipe.unet.attn_processors.items():
        if hasattr(processor, 'state_dict'):
            for k, v in processor.state_dict().items():
                lora_state_dict[f"{name}.{k}"] = v

    os.makedirs(output_dir, exist_ok=True)
    torch.save(lora_state_dict, os.path.join(output_dir, "teacher_lora_weights_clean.bin"))
    print("Poids LoRA sauvegardés dans :", os.path.join(output_dir, "teacher_lora_weights_clean.bin"))

    # Affichage de la courbe de convergence
    smoothed = smooth_curve(loss_history)
    plot_loss_curve(smoothed)

    return loss_history
