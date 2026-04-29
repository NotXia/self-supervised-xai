import os
from pathlib import Path
import requests
import uuid
import numpy as np
from copy import copy


API_KEY = open("../.tally-key", "r").read()
ASSETS_URL = open("../.assets-url", "r").read()
METHODS = ["ours", "saliency", "int_gradients", "deeplift", "gradient_shap"]
curr_page_idx = 0


def page_break_block():
    global curr_page_idx
    curr_page_idx += 1
    return {
        'uuid': f'{uuid.uuid4()}',
        'type': 'PAGE_BREAK',
        'groupUuid': f'{uuid.uuid4()}',
        'groupType': 'PAGE_BREAK',
        'payload': {
            "index": curr_page_idx-1
        },
    }

def divider_block():
    return {
        'uuid': f'{uuid.uuid4()}',
        'type': 'DIVIDER',
        'groupUuid': f'{uuid.uuid4()}',
        'groupType': 'DIVIDER',
        'payload': {},
    }

def title_block(title):
    return {
        'uuid': f'{uuid.uuid4()}',
        'type': 'TITLE',
        'groupUuid': f'{uuid.uuid4()}',
        'groupType': 'TITLE',
        'payload': {
            'html': title,
        },
    }

def text_block(text):
    return {
        'uuid': f'{uuid.uuid4()}',
        'type': 'TEXT',
        'groupUuid': f'{uuid.uuid4()}',
        'groupType': 'TEXT',
        'payload': {
            'html': text
        },
    }

def h1_block(text):
    return {
        'uuid': f'{uuid.uuid4()}',
        'type': 'HEADING_1',
        'groupUuid': f'{uuid.uuid4()}',
        'groupType': 'HEADING_1',
        'payload': {
            'html': text
        },
    }

def h2_block(text):
    return {
        'uuid': f'{uuid.uuid4()}',
        'type': 'HEADING_2',
        'groupUuid': f'{uuid.uuid4()}',
        'groupType': 'HEADING_2',
        'payload': {
            'html': text
        },
    }

def linear_scale_block(name):
    return {
        'uuid': f'{uuid.uuid4()}',
        'type': 'LINEAR_SCALE',
        'groupUuid': f'{uuid.uuid4()}',
        'groupType': 'LINEAR_SCALE',
        'payload': {
            "isRequired": False,
            "hasDefaultAnswer": False,
            "start": 1,
            "end": 5,
            "step": 1,
            "hasLeftLabel": True,
            "leftLabel": "Poor",
            "hasCenterLabel": True,
            "centerLabel": "Acceptable",
            "hasRightLabel": True,
            "rightLabel": "Excellent",
            "name": name
        },
    }

def image_block(url):
    return {
        'uuid': f'{uuid.uuid4()}',
        'type': 'EMBED',
        'groupUuid': f'{uuid.uuid4()}',
        'groupType': 'EMBED',
        'payload': {
            "type": "image/*",
            "provider": "Image",
            "title": "Sample",
            "inputUrl": url,
            "display": {
                "url": url
            },
        },
    }

def audio_block(url):
    return {
        'uuid': f'{uuid.uuid4()}',
        'type': 'EMBED',
        'groupUuid': f'{uuid.uuid4()}',
        'groupType': 'EMBED',
        'payload': {
            "type": "audio/*",
            "provider": "Audio",
            "title": "Sample",
            "inputUrl": url,
            "display": {
                "url": url
            },
        },
    }

def load_images(path, rng):
    out_blocks = []
    instructions = (
        "The task of the dataset is object classification. Each image contains a single object to classify.<br/>"
        "The heatmap overlay indicates which regions of the image the model focused on. "
        "Warmer colors indicate higher relevance. "
        "Rate how well the highlighted region justifies the predicted class and matches what you would consider the most relevant part of the image.<br/><br/>"
        "Rating scale:<br/>"
        "1 = Poor: the explanation does not seem related to the predicted class and/or does not match my intuition<br/>"
        "2 = Fair: the explanation is weakly related to the predicted class or only partially matches my intuition<br/>"
        "3 = Acceptable: the explanation is reasonably related to the predicted class and somewhat matches my intuition<br/>"
        "4 = Good: the explanation clearly highlights relevant parts and mostly matches my intuition<br/>"
        "5 = Excellent: the explanation is highly faithful to the predicted class and strongly matches my intuition"
    )

    out_blocks.append( h1_block("Image samples") )
    out_blocks.append( text_block(
        (
            "This section contains 10 random samples of an image classification dataset. <br/>"
            + "We will present the prediction of the model and the original image. Then, 5 images overlayed with a heatmap will follow. <br/>"
            + "For each one, you will be able to give a preference score from 1 to 5. <br/><br/>"
            + "Detailed istructions are as follows (they will be visible in each page): <br/>"
            + instructions
        )
    ) )
    out_blocks.append( page_break_block() )

    for sample_dir in os.listdir(path):
        base_dir = os.path.join(path, sample_dir)
        rel_dir = os.path.join(Path(path).name, sample_dir)
        prediction = open(os.path.join(base_dir, "label.txt"), "r").read()

        # out_blocks.append( title_block("Image sample") )
        out_blocks.append( text_block(instructions) )
        out_blocks.append( divider_block() )
        out_blocks.append( h1_block(f"Input image") )
        out_blocks.append( h2_block(f"The model predicted: {prediction.upper()}") )
        out_blocks.append( image_block(os.path.join(ASSETS_URL, rel_dir, "original.png")) )
        out_blocks.append( divider_block() )
        out_blocks.append( divider_block() )
        out_blocks.append( h1_block(f"Explanations") )
        out_blocks.append( divider_block() )

        methods = copy(METHODS)
        rng.shuffle(methods)
        for method in methods:
            out_blocks.append( image_block(os.path.join(ASSETS_URL, rel_dir, f"{method}.png")) )
            out_blocks.append( linear_scale_block(f"image-{sample_dir}-{method}") )
            out_blocks.append( divider_block() )
            out_blocks.append( divider_block() )

        out_blocks.append( page_break_block() )

    return out_blocks


def load_texts(path, rng):
    out_blocks = []
    instructions = (
        "The task of the dataset is sentiment classification from social media posts.<br/>"
        "The highlighted words indicate which tokens the model focused on to make its prediction. "
        "Words with a dark red background are considered highly relevant. "
        "Rate how well the highlighted words justify the predicted class and match what you would consider relevant in the sentence.<br/><br/>"
        "Rating scale:<br/>"
        "1 = Poor: the explanation does not seem related to the predicted class and/or does not match my intuition<br/>"
        "2 = Fair: the explanation is weakly related to the predicted class or only partially matches my intuition<br/>"
        "3 = Acceptable: the explanation is reasonably related to the predicted class and somewhat matches my intuition<br/>"
        "4 = Good: the explanation clearly highlights relevant parts and mostly matches my intuition<br/>"
        "5 = Excellent: the explanation is highly faithful to the predicted class and strongly matches my intuition"
    )

    out_blocks.append( h1_block("Text samples") )
    out_blocks.append( text_block(
        (
            "This section contains 10 random samples of a text classification dataset. <br/>"
            + "We will present the prediction of the model and the original document. Then, 5 documents with highlighted words will follow. <br/>"
            + "For each one, you will be able to give a preference score from 1 to 5. <br/><br/>"
            + "Detailed istructions are as follows (they will be visible in each page): <br/>"
            + instructions
        )
    ) )
    out_blocks.append( page_break_block() )

    for sample_dir in os.listdir(path):
        base_dir = os.path.join(path, sample_dir)
        rel_dir = os.path.join(Path(path).name, sample_dir)
        prediction = open(os.path.join(base_dir, "label.txt"), "r").read()

        # out_blocks.append( title_block("Text sample") )
        out_blocks.append( text_block(instructions) )
        out_blocks.append( divider_block() )
        out_blocks.append( h1_block(f"Input text") )
        out_blocks.append( h2_block(f"The model predicted: {prediction.upper()}") )
        out_blocks.append( image_block(os.path.join(ASSETS_URL, rel_dir, "original.png")) )
        out_blocks.append( divider_block() )
        out_blocks.append( divider_block() )
        out_blocks.append( h1_block(f"Explanations") )
        out_blocks.append( divider_block() )

        methods = copy(METHODS)
        rng.shuffle(methods)
        for method in methods:
            out_blocks.append( image_block(os.path.join(ASSETS_URL, rel_dir, f"{method}.png")) )
            out_blocks.append( linear_scale_block(f"text-{sample_dir}-{method}") )
            out_blocks.append( divider_block() )
            out_blocks.append( divider_block() )

        out_blocks.append( page_break_block() )

    return out_blocks


def load_audios(path, rng):
    out_blocks = []
    instructions = (
        "The task of the dataset is spoken word detection. Each audio track contains noise and one spoken word.<br/>"
        "Listen to the original audio clip first, then listen to each explanation. "
        "Each explanation is a version of the audio where the most relevant parts have been preserved according to the model. "
        "Rate how well the preserved audio corresponds to the predicted class and matches what you would consider the most relevant part of the clip.<br/><br/>"
        "Rating scale:<br/>"
        "1 = Poor: the explanation does not seem related to the predicted class and/or does not match my intuition<br/>"
        "2 = Fair: the explanation is weakly related to the predicted class or only partially matches my intuition<br/>"
        "3 = Acceptable: the explanation is reasonably related to the predicted class and somewhat matches my intuition<br/>"
        "4 = Good: the explanation clearly highlights relevant parts and mostly matches my intuition<br/>"
        "5 = Excellent: the explanation is highly faithful to the predicted class and strongly matches my intuition"
    )

    out_blocks.append( h1_block("Audio samples") )
    out_blocks.append( text_block(
        (
            "This section contains 10 random samples of an audio classification dataset. <br/>"
            + "We will present the prediction of the model and the original audio track. Then, 5 audio tracks where the most relevant parts are preserved will follow. <br/>"
            + "For each one, you will be able to give a preference score from 1 to 5. <br/><br/>"
            + "Detailed istructions are as follows (they will be visible in each page): <br/>"
            + instructions
        )
    ) )
    out_blocks.append( page_break_block() )

    for sample_dir in os.listdir(path):
        base_dir = os.path.join(path, sample_dir)
        rel_dir = os.path.join(Path(path).name, sample_dir)
        prediction = open(os.path.join(base_dir, "label.txt"), "r").read()

        # out_blocks.append( title_block("Audio sample") )
        out_blocks.append( text_block(instructions) )
        out_blocks.append( divider_block() )
        out_blocks.append( h1_block(f"Input audio") )
        out_blocks.append( h2_block(f"The model predicted: {prediction.upper()}") )
        out_blocks.append( audio_block(os.path.join(ASSETS_URL, rel_dir, "original.wav")) )
        out_blocks.append( divider_block() )
        out_blocks.append( divider_block() )
        out_blocks.append( h1_block(f"Explanations") )
        out_blocks.append( divider_block() )

        methods = copy(METHODS)
        rng.shuffle(methods)
        for method in methods:
            out_blocks.append( audio_block(os.path.join(ASSETS_URL, rel_dir, f"{method}.wav")) )
            out_blocks.append( linear_scale_block(f"audio-{sample_dir}-{method}") )
            out_blocks.append( divider_block() )
            out_blocks.append( divider_block() )

        out_blocks.append( page_break_block() )

    return out_blocks


def load_multimodal(path, rng):
    out_blocks = []
    instructions = (
        "The task of the dataset is caption verification. Each image is paired with a caption that can be correct or incorrect.<br/>"
        "The explanation highlights relevant words in the text and relevant regions in the image. "
        "Rate how well the combination of both justifies the predicted class and matches your intuition about what is relevant across both modalities.<br/><br/>"
        "Rating scale:<br/>"
        "1 = Poor: the explanation does not seem related to the predicted class and/or does not match my intuition<br/>"
        "2 = Fair: the explanation is weakly related to the predicted class or only partially matches my intuition<br/>"
        "3 = Acceptable: the explanation is reasonably related to the predicted class and somewhat matches my intuition<br/>"
        "4 = Good: the explanation clearly highlights relevant parts and mostly matches my intuition<br/>"
        "5 = Excellent: the explanation is highly faithful to the predicted class and strongly matches my intuition"
    )

    out_blocks.append( h1_block("Multimodal samples") )
    out_blocks.append( text_block(
        (
            "This section contains 10 random samples of a text-image classification dataset. <br/>"
            + "We will present the prediction of the model and the original text-image pair. Then, 5 text-image pairs with highlighted words and heatmap overlay will follow. <br/>"
            + "For each one, you will be able to give a preference score from 1 to 5. <br/><br/>"
            + "Detailed istructions are as follows (they will be visible in each page): <br/>"
            + instructions
        )
    ) )
    out_blocks.append( page_break_block() )

    for sample_dir in os.listdir(path):
        base_dir = os.path.join(path, sample_dir)
        rel_dir = os.path.join(Path(path).name, sample_dir)
        prediction = open(os.path.join(base_dir, "label.txt"), "r").read()

        # out_blocks.append( title_block("Multimodal sample") )
        out_blocks.append( text_block(instructions) )
        out_blocks.append( divider_block() )
        out_blocks.append( h1_block(f"Input text-image") )
        out_blocks.append( h2_block(f"The model predicted: {prediction.upper()}") )
        out_blocks.append( image_block(os.path.join(ASSETS_URL, rel_dir, "original.png")) )
        out_blocks.append( divider_block() )
        out_blocks.append( divider_block() )
        out_blocks.append( h1_block(f"Explanations") )
        out_blocks.append( divider_block() )

        methods = copy(METHODS)
        rng.shuffle(methods)
        for method in methods:
            out_blocks.append( image_block(os.path.join(ASSETS_URL, rel_dir, f"{method}.png")) )
            out_blocks.append( linear_scale_block(f"multimodal-{sample_dir}-{method}") )
            out_blocks.append( divider_block() )
            out_blocks.append( divider_block() )

        out_blocks.append( page_break_block() )

    return out_blocks
        

if __name__ == "__main__":
    form_blocks = [
        {
            'uuid': f'{uuid.uuid4()}',
            'type': 'FORM_TITLE',
            'groupUuid': f'{uuid.uuid4()}',
            'groupType': 'TEXT',
            'payload': {
                'html': 'Evaluation of Attribution Maps',
            },
        },
        title_block("Introduction"),
        text_block(
            (
                "In this study, you will evaluate explanations produced by AI models across four modalities: text, image, audio, and multimodal. "
                "For each question, you are shown an input and the class predicted by an AI classifier. "
                "You are then shown 4 explanations, each produced by a different method. "
                "Your task is to rate each explanation from 1 to 5 based on two criteria considered together: <br/>"
                "  (1) how well it highlights the parts of the input that justify the predicted class, and <br/>"
                "  (2) how well it matches your own intuition about what is relevant for that class. <br/>"
                "For each modality, a short specific instruction will be provided before the corresponding questions to clarify what you are looking at. "
                "There are no right or wrong answers; we are interested in your perception."
            )
        ),
        page_break_block(),
    ]

    rng = np.random.default_rng(42)

    form_blocks += load_images("../samples-user-study/image", rng)
    form_blocks += load_texts("../samples-user-study/text", rng)
    form_blocks += load_audios("../samples-user-study/audio", rng)
    form_blocks += load_multimodal("../samples-user-study/multimodal", rng)

    form_blocks += [
        h1_block("Thank you!"),
        text_block("Thank you for taking your time to fill this form!")
    ]

    response = requests.post(
        'https://api.tally.so/forms', 
        headers = {
            'Authorization': f'Bearer {API_KEY}',
            'Content-Type': 'application/json',
        }, 
        json = {
            'status': 'PUBLISHED',
            'blocks': form_blocks,
        }
    )

    print(response.text)