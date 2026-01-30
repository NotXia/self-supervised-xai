import nltk
nltk.download('punkt_tab')


from .image.mnist import MNISTDataset
from .image.cifar10 import CIFAR10Dataset
from .image.imagenette import ImagenetteDataset

from .text.tweet_sentiment import TweetSentimentDataset
from .text.pubhealth import PubHealthDataset
from .text.imdb import IMDBDataset
from .text.politifact import PolitifactDataset

from .text_image.flickr8k import Flickr8kDataset
from .text_image.hateful_memes import HatefulMemesDataset

from .qa.race import RACEDataset
from .qa.quail import QuAILDataset
from .qa.swag import SWAGDataset
from .qa.figureqa import FigureQADataset

from .claim_verification.sciver import SciVerSimpleDataset
from .sciclaimeval import SciClaimEvalDataset
from .scitabalign import SciTabAlignDataset

from .utils import *



DATASETS_MAP = {
    "mnist": MNISTDataset,
    "cifar10": CIFAR10Dataset,
    "imagenette": ImagenetteDataset,
    
    "imdb": IMDBDataset,
    "tweet-sentiment": TweetSentimentDataset,
    "politifact": PolitifactDataset,
    # "pubhealth": PubHealthDataset,

    "flickr8k": Flickr8kDataset,
    "hateful-memes": HatefulMemesDataset,

    "sciver-simple": SciVerSimpleDataset,
    # "sciclaimeval": SciClaimEvalDataset,
    # "scitabalign": SciTabAlignDataset,

    "race": RACEDataset,
    "quail": QuAILDataset,
    "swag": SWAGDataset,
    "figureqa": FigureQADataset,
}


DATASETS = DATASETS_MAP.keys()


def get_dataset(
    dataset_name: str,
    data_dir: str,
    splits = ["train", "validation", "test"],
    seed = 42,
    *args,
    **kwargs
):
    DatasetClass = DATASETS_MAP[dataset_name]

    return (
        *[
            DatasetClass(split, data_dir, seed, *args, **kwargs)
            for split in splits
        ], 
        DatasetClass.num_classes
    )