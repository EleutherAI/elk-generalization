from binary_operation_dataset import (
    AdditionDataset,
    ModularAdditionDataset,
    MultiplicationDataset,
    SubtractionDataset,
)
from books_dataset import AuthorsDataset
from cities_dataset import CapitalsDataset, HemisphereDataset, PopulationDataset
from nli_dataset import NliDataset
from sciq_dataset import SciQDataset
from sentiment_dataset import SentimentDataset
from unary_operation_dataset import SquaringDataset

ds_classes = [
    (SentimentDataset, 8000),
    (NliDataset, 4000),
    (SciQDataset, 4000),
    (PopulationDataset, 4000),
    (CapitalsDataset, 2000),
    (HemisphereDataset, 4000),
    (AuthorsDataset, 4000),
    (AdditionDataset, 8000),
    (SubtractionDataset, 8000),
    (MultiplicationDataset, 8000),
    (ModularAdditionDataset, 8000),
    (SquaringDataset, 8000),
]

if __name__ == "__main__":
    for ds_class, n_val_test in ds_classes:
        pythia_suite = [
            "EleutherAI/pythia-160m-v0",
            "EleutherAI/pythia-410m",
            "EleutherAI/pythia-1b",
            "EleutherAI/pythia-1.4b",
            "EleutherAI/pythia-2.8b",
            "EleutherAI/pythia-6.9b",
            "EleutherAI/pythia-12b",
        ][::-1]

        models = (
            pythia_suite
            if ds_class in {SentimentDataset, NliDataset, SciQDataset}
            else []
        )

        ds = ds_class(working_dir="weak_lm_datasets", verbose=True)
        print("Creating dataset", ds.name)
        ds.save_quirky_dataset(
            difficulty_model_names=models,
            push_to_hub=True,
            n_train=-1,
            n_val=n_val_test,
            n_test=n_val_test,
        )
