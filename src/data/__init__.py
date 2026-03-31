from .image.mnist import MNISTDataset
from .image.cifar10 import CIFAR10Dataset
from .image.imagenette import ImagenetteDataset
from .image.oxford_pet import OxfordPetDataset

from .text.tweet_sentiment import TweetSentimentDataset
from .text.imdb import IMDBDataset
from .text.politifact import PolitifactDataset
from .text.hatexplain import HateXplainDataset

from .text_image.flickr8k import Flickr8kDataset
from .text_image.hateful_memes import HatefulMemesDataset
from .text_image.snli_ve import SNLIVEDataset

from .audio.tut_urban import TUTUrbanDataset




DATASETS_MAP = {
    "mnist": MNISTDataset,
    "cifar10": CIFAR10Dataset,
    "imagenette": ImagenetteDataset,
    "oxford-pet": OxfordPetDataset,
    
    "imdb": IMDBDataset,
    "tweet-sentiment": TweetSentimentDataset,
    "politifact": PolitifactDataset,
    "fever": FEVERDataset,
    "movie-rationales": MovieRationalesDataset,
    "cos-e": CoSEDataset,
    "hatexplain": HateXplainDataset,

    "flickr8k": Flickr8kDataset,
    "hateful-memes": HatefulMemesDataset,
    "snli-ve": SNLIVEDataset,

    "tut-urban": TUTUrbanDataset
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