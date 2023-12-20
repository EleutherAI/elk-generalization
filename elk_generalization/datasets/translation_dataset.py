import os
from collections import defaultdict
from multiprocessing.pool import ThreadPool

import numpy as np
import pandas as pd
from datasets import Dataset
from tqdm.auto import tqdm
from translate import Translator

from .quirky_dataset import QuirkyDataset


# Sourced from https://www.kaggle.com/datasets/rtatman/english-word-frequency
class TranslationDataset(QuirkyDataset):
    quirky_template = (
        'Is "{translation}" the Spanish translation of "{english}"? {character}:'
    )
    quirky_choices = (" No", " Yes")

    def __init__(
        self,
        source_path: str = "data/unigram_freq.csv",
        cache_path: str = "data/unigram_freq_translated.csv",
        **kwargs,
    ):
        self.source_loc = source_path
        self.cache_loc = cache_path
        super().__init__(**kwargs)

    def _load(self) -> Dataset:
        if os.path.exists(self.cache_loc):
            df = pd.read_csv(self.cache_loc)
        else:
            df = pd.read_csv(self.source_loc)[:18_000]
            df["word"] = df["word"].astype(str)
            assert sum(df["count"].isna()) == 0

            translations = defaultdict(list)
            for target_lang in ["es", "pt"]:
                translator = Translator(to_lang=target_lang)

                with ThreadPool(8) as p:
                    translations[target_lang] = list(
                        tqdm(
                            p.imap(translator.translate, df["word"].values),
                            total=len(df),
                            disable=not self.verbose,
                            desc=f"Translating to {target_lang}",
                        )
                    )

            df.rename(columns={"word": "en"}, inplace=True)
            df["es"] = translations["es"]
            df["pt"] = translations["pt"]

        # distractors for each language come from the same language,
        # and are just a similarly common translation
        df = df.sort_values("count", ascending=False).reset_index(drop=True)
        df["es_distractor"] = df["es"].shift(1)
        df["pt_distractor"] = df["pt"].shift(1)
        df = df.dropna()  # drop the first row

        # unpivot table so that there's 4x as many examples: [es, pt] x [true, false]
        df = df.melt(
            id_vars=["en", "es", "pt", "count"],
            value_vars=["es", "pt", "es_distractor", "pt_distractor"],
            var_name="lang_x_distractor",
            value_name="translation",
        )

        df["alice_label"] = df["es"] == df["translation"]
        df["bob_label"] = df["pt"] == df["translation"]
        df["difficulty"] = -df["count"].apply(np.log10)

        df.to_csv(self.cache_loc, index=False)

        return Dataset.from_pandas(df)

    def _generate_base_dataset(
        self,
        n_total,
        difficulty_model_names: list[str],
    ) -> tuple[Dataset, dict]:
        return self.dataset.select(range(n_total)), dict()

    def _quirky_map_function(self, examples):
        results = defaultdict(list)
        batch_size = len(examples["difficulty"])
        for i in range(batch_size):
            for character in ["Alice", "Bob"]:
                statement = self.quirky_template.format(
                    english=examples["en"][i],
                    translation=examples["translation"][i],
                    character=character,
                )
                results["statement"].append(statement)
                results["choices"].append(self.quirky_choices)
                results["character"].append(character)
                results["label"].append(examples[f"{character.lower()}_label"][i])
                results["alice_label"].append(examples["alice_label"][i])
                results["bob_label"].append(examples["bob_label"][i])
                results["difficulty"].append(examples["difficulty"][i])
        return results
