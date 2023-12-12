from elk_generalization.datasets.sciq_dataset import SciQDataset
from elk_generalization.datasets.race_dataset import RaceDataset
from elk_generalization.datasets.popqa_dataset import PopQADataset


dataset_types = [SciQDataset, PopQADataset]
# dataset_types = [SciQDataset, RaceDataset, PopQADataset][::-1]
for dataset_type in dataset_types:
    ds = dataset_type("weak_lm_datases")

    print(f"Dataset: {dataset_type.__name__}")

    pythia_suite = [f"EleutherAI/pythia-{size}" for size in ["14m", "70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b"]]
    # pythia_suite = [f"EleutherAI/pythia-{size}" for size in ["14m", "70m", "160m"]] #, "410m", "1b", "1.4b", "2.8b", "6.9b", "12b"]]
    ds.generate_quirky_dataset(weak_model_name="EleutherAI/pythia-410m", difficulty_model_names=pythia_suite, push_to_hub=True, n_train=-1, n_val=1000, n_test=1000, verbose=True)
