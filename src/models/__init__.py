from .ImageClassificationModel import ImageClassificationModel, TRAINING_PHASES
from .TextClassificationModel import TextClassificationModel, TRAINING_PHASES
from .MMClassificationModel import MMClassificationModel, TRAINING_PHASES

from .utils.utils import *
from .utils.masker import *



MODELS_MAP = {
    "text-classifier": TextClassificationModel,
    "image-classifier": ImageClassificationModel,
    "mm-classifier": MMClassificationModel,
}

def get_model(model_name, checkpoint_path=None, *args, **kwargs):
    ModelClass = MODELS_MAP[model_name]

    if checkpoint_path is not None:
        return ModelClass.load_from_checkpoint(
            checkpoint_path,
            map_location = "cpu",
            *args, **kwargs
        )
    else:
        return ModelClass(*args, **kwargs)