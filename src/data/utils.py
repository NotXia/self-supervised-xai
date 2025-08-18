from .pubhealth import PubHealthDataset
from .sciver import SciVerDataset


DATASETS_MAP = {
    "pubhealth": (PubHealthDataset, 4),
    "sciver": (SciVerDataset, 2),
}

DATASETS = DATASETS_MAP.keys()

def get_dataset(
    dataset_name: str,
    data_dir: str,
    sep_tok: str,
    splits = ["train", "validation", "test"],
    seed = 42
):
    DataClass, num_classes = DATASETS_MAP[dataset_name]

    return (
        *[
            DataClass(split, sep_tok, data_dir, seed)
            for split in splits
        ], 
        num_classes
    )
