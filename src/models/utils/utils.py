from transformers import AutoConfig


def get_sep_token(model_card):
    match model_card:
        case "Qwen/Qwen3-Embedding-0.6B":
            return "<|endoftext|>", 151643
        case "Qwen/Qwen3-Embedding-4B":
            return "<|endoftext|>", 151643
        case _:
            raise RuntimeError("Unhandled model")



def get_vit_config(model_card):
    config = AutoConfig.from_pretrained(model_card)
    return {
        "hidden_size": config.pooler_output_size,
        "in_resolution": ( int(config.image_size), int(config.image_size) ),
        "out_resolution": ( int(config.image_size/config.patch_size), int(config.image_size/config.patch_size) )
    }
