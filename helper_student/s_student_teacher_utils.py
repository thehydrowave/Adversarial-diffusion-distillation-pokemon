# student_teacher_utils.py
import torch
from diffusers.models.attention_processor import LoRAAttnProcessor

def inject_lora_modules(unet):
    cross_attention_dim = unet.config.cross_attention_dim
    attention_dims = {
        "mid_block": unet.config.block_out_channels[-1],
        "up_blocks.3": unet.config.block_out_channels[0],
        "up_blocks.2": unet.config.block_out_channels[1],
        "up_blocks.1": unet.config.block_out_channels[2],
        "up_blocks.0": unet.config.block_out_channels[3],
        "down_blocks.0": unet.config.block_out_channels[0],
        "down_blocks.1": unet.config.block_out_channels[1],
        "down_blocks.2": unet.config.block_out_channels[2],
        "down_blocks.3": unet.config.block_out_channels[3],
    }

    attn_processors = {}
    for name, processor in unet.attn_processors.items():
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

    unet.set_attn_processor(attn_processors)
    print(f"LoRA injecté dans {sum(isinstance(p, LoRAAttnProcessor) for p in attn_processors.values())} couches.")

def load_lora_weights_from_bin(unet, lora_path):
    print(f"Chargement des poids LoRA depuis : {lora_path}")
    lora_weights = torch.load(lora_path)
    updated = 0
    for full_key, module in unet.attn_processors.items():
        if hasattr(module, 'load_state_dict'):
            prefix = full_key + "."
            sub_state_dict = {k[len(prefix):]: v for k, v in lora_weights.items() if k.startswith(prefix)}
            if sub_state_dict:
                try:
                    module.load_state_dict(sub_state_dict, strict=False)
                    updated += 1
                except Exception as e:
                    print(f"Erreur pour {full_key} : {e}")
    print(f" {updated}/{len(unet.attn_processors)} modules LoRA mis à jour.")
