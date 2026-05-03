import os
from pathlib import Path
import requests
import uuid
import numpy as np
from copy import copy


API_KEY = open("../.tally-key", "r").read()
BASE_URL = open("../.assets-url", "r").read()
ASSETS_URL = os.path.join(BASE_URL, "samples-user-study")
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
            "isRequired": True,
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
        "The task of the dataset is <strong>object classification</strong>. Each image contains a single object to classify.<br/>"
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
            "This section contains <strong>10 random samples</strong> of an <strong>image classification</strong> dataset. <br/>"
            + "We will present the prediction of the model and the original image. Then, <strong>5 images overlayed with a heatmap</strong> will follow. <br/>"
            + "For each one, you will be able to give a <strong>preference score from 1 to 5</strong>. <br/><br/>"
            + "Detailed instructions are as follows (they will be visible in each page): <br/>"
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
        "The task of the dataset is <strong>sentiment classification</strong> from movie reviews.<br/>"
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
            "This section contains <strong>10 random samples</strong> of a <strong>text classification</strong> dataset. <br/>"
            + "We will present the prediction of the model and the original document. Then, <strong>5 documents with highlighted words</strong> will follow. <br/>"
            + "For each one, you will be able to give a <strong>preference score from 1 to 5</strong>. <br/><br/>"
            + "Detailed instructions are as follows (they will be visible in each page): <br/>"
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
        "The task of the dataset is <strong>spoken word detection</strong>. Each audio track contains noise and one spoken word.<br/>"
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
            "This section contains <strong>10 random samples</strong> of an <strong>audio classification</strong> dataset. <br/>"
            + "We will present the prediction of the model and the original audio track. Then, <strong>5 audio tracks where the most relevant parts are preserved</strong> will follow. <br/>"
            + "For each one, you will be able to give a <strong>preference score from 1 to 5</strong>. <br/><br/>"
            + "Detailed instructions are as follows (they will be visible in each page): <br/>"
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
        "The task of the dataset is <strong>caption verification</strong>. Each image is paired with a caption that can be correct or incorrect.<br/>"
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
            "This section contains <strong>10 random samples</strong> of a <strong>text-image classification</strong> dataset. <br/>"
            + "We will present the prediction of the model and the original text-image pair. Then, <strong>5 text-image pairs with highlighted words and heatmap overlay</strong> will follow. <br/>"
            + "For each one, you will be able to give a <strong>preference score from 1 to 5</strong>. <br/><br/>"
            + "Detailed instructions are as follows (they will be visible in each page): <br/>"
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
                "In this study, you will evaluate explanations produced by AI models across four modalities: <strong>text, image, audio, and multimodal</strong>. "
                "For each question, you are shown an <strong>input and the class predicted by an AI classifier</strong>. "
                "You are then shown <strong>5 explanations</strong>, each produced by a different method. "
                "Your task is to <strong>rate each explanation from 1 to 5</strong> based on two criteria considered together: <br/>"
                "  (1) how well it highlights the parts of the input that justify the predicted class, and <br/>"
                "  (2) how well it matches your own intuition about what is relevant for that class. <br/>"
                "For each modality, a short specific instruction will be provided before the corresponding questions to clarify what you are looking at. "
                "<strong>There are no right or wrong answers</strong>; we are interested in your perception."
            )
        ),
        page_break_block(),
        title_block("Check that everything works"),
        text_block(
            (
                "To make sure everything is working, check if you can see the picture and listen to the audio track. If it does not work, try to change browser."
            )
        ),
        image_block(os.path.join("https://picsum.photos/536/354")),
        audio_block(os.path.join(BASE_URL, "samples/luma/sample_0/original.wav")),
        page_break_block(),
    ]

    rng = np.random.default_rng(42)

    form_blocks += load_images("../samples-user-study/image", rng)
    form_blocks += load_texts("../samples-user-study/text", rng)
    form_blocks += load_audios("../samples-user-study/audio", rng)
    form_blocks += load_multimodal("../samples-user-study/multimodal", rng)

    form_blocks += [
        h1_block("Thank you!"),
        text_block("Thank you for taking your time to fill this form!"),
        image_block(os.path.join("https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExcXdobXNueTh0bXY4aW8ybGVqemcxdGl0YnJha2l3NnQxYWwxMXA3eCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/PGSnPf27XvtTXGTKIG/giphy.gif")),
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