# model_utils.py
import torch
from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import LoRAAttnProcessor

# model_utils.py
import torch
from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import LoRAAttnProcessor

def load_pipeline(model_id):
    print(f"Chargement du modèle Stable Diffusion depuis : {model_id}")
    
    try:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe.to("cuda")
        pipe.enable_attention_slicing()

        print("Pipeline chargé avec succès.")
        print(f"Modèle utilisé : {pipe.__class__.__name__}")
        print(f"UNet type : {pipe.unet.__class__.__name__}")
        print(f"Tokenizer : {pipe.tokenizer.__class__.__name__}")
        print(f"Taille des embeddings texte : {pipe.text_encoder.config.hidden_size}")
        print(f"Nombre de steps de diffusion : {pipe.scheduler.config.num_train_timesteps}")
        print(f"Type de scheduler : {pipe.scheduler.__class__.__name__}")
    
    except Exception as e:
        print("Erreur lors du chargement du pipeline :", str(e))
        raise e

    return pipe


def apply_lora(pipe):
    cross_attention_dim = pipe.unet.config.cross_attention_dim
    attention_dims = {
        "mid_block": pipe.unet.config.block_out_channels[-1],
        "up_blocks.3": pipe.unet.config.block_out_channels[0],
        "up_blocks.2": pipe.unet.config.block_out_channels[1],
        "up_blocks.1": pipe.unet.config.block_out_channels[2],
        "up_blocks.0": pipe.unet.config.block_out_channels[3],
        "down_blocks.0": pipe.unet.config.block_out_channels[0],
        "down_blocks.1": pipe.unet.config.block_out_channels[1],
        "down_blocks.2": pipe.unet.config.block_out_channels[2],
        "down_blocks.3": pipe.unet.config.block_out_channels[3],
    }

    attn_processors = {}
    for name, processor in pipe.unet.attn_processors.items():
        for block_key in attention_dims:
            if name.startswith(block_key):
                hidden_size = attention_dims[block_key]
                is_cross_attention = name.endswith("attn2.processor")
                attn_processors[name] = LoRAAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim if is_cross_attention else None
                )
                break
        else:
            attn_processors[name] = processor

    pipe.unet.set_attn_processor(attn_processors)

    lora_params = []
    count = 0
    for name, processor in pipe.unet.attn_processors.items():
        if isinstance(processor, LoRAAttnProcessor):
            count += 1
            for param in processor.parameters():
                param.requires_grad = True
                lora_params.append(param)

    print(f"LoRA appliqué à {count} couches.")
    print(f"Nombre total de paramètres LoRA entraînables : {sum(p.numel() for p in lora_params)}")
    return lora_params
