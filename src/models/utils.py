def get_sep_token(model_card):
    match model_card:
        case "Qwen/Qwen3-Embedding-0.6B":
            return "<|endoftext|>", 151643
        case _:
            raise RuntimeError("Unhandled model")