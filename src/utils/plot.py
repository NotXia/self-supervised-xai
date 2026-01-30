import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colormaps



def _is_chw(image):
    return (image.shape[0] == 1) or (image.shape[0] == 3)


def plot_image_attribution(image, attribution, plot_complete=True, alpha=0.5, figsize=(8, 3)):
    image = np.asarray(image)
    attribution = np.asarray(attribution)

    image = image.transpose(1, 2, 0) if _is_chw(image) else image
    attribution = attribution.transpose(1, 2, 0) if _is_chw(attribution) else attribution

    if plot_complete:
        # ax = plt.gca()
        fig, axs = plt.subplots(1, 3, layout="constrained", figsize=figsize)

        axs[0].imshow(image)
        axs[0].set_xticks([]); axs[0].set_yticks([])

        axs[1].imshow(image)
        axs[1].imshow(attribution, alpha=alpha)
        axs[1].set_xticks([]); axs[1].set_yticks([])

        im = axs[2].imshow(attribution)
        axs[2].set_xticks([]); axs[2].set_yticks([])
        plt.colorbar(im, ax=axs[2], shrink=0.7, location="right")
    else:
        plt.imshow(image)
        plt.imshow(attribution, alpha=alpha)
        plt.xticks([]); plt.yticks([])
        plt.colorbar(im, ax=plt.gca(), shrink=0.7, location="right")



def decode_text_with_scores(tokens, weights, tokenizer, eos_tok):
    text = []
    scores = []
    curr_word = []

    for t, w in zip(tokens, weights):
        tok = tokenizer.decode(t)
        if tok == eos_tok: break

        if (len(curr_word) != 0) and (tok[0] == ' '):
            text.append("".join([t for t, w in curr_word]).strip())
            scores.append(sum([w for t, w in curr_word]) / len(curr_word))
            curr_word = []
        curr_word.append((tok, w))
        
    if len(curr_word) != 0:
        text.append("".join([t for t, w in curr_word]).strip())
        scores.append(sum([w for t, w in curr_word]) / len(curr_word))

    return text, scores


def highlighter_html(word, score):
    color = colormaps["OrRd"](round((score) * 255))
    color = [round(c*255) for c in color]
    return f"<span style='color: {'white' if score > 0.75 else 'black'}; background-color: rgb({color[0]} {color[1]} {color[2]})'>{word}</span>"


def display_text_attribution(tokens, weights, tokenizer, eos_tok):
    from IPython.display import display, HTML

    text, scores = decode_text_with_scores(tokens, weights, tokenizer, eos_tok)
    display(HTML(
        ' '.join([highlighter_html(t, s) for t, s in zip(text, scores)])
    ))