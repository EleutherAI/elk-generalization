from collections import defaultdict

import numpy as np
import pandas as pd
from datasets import Dataset

from .quirky_dataset import QuirkyDataset


# Sourced from https://www.kaggle.com/datasets/viswanathanc/world-cities-datasets/
class CitiesDataset(QuirkyDataset):
    quirky_template = (
        "Does {character} think that {city} is the capital of {admin_name}, {country}?"
    )
    quirky_choices = (" No", " Yes")

    def __init__(self, source_path: str = "data/worldcities.csv", **kwargs):
        self.source_loc = source_path
        super().__init__(**kwargs)

    def _load(self) -> Dataset:
        df = pd.read_csv(self.source_loc)
        # we want to get the model to state whether the city
        # is the capital of the admin_name
        df = df[
            (df["city"] != df["admin_name"])
            & ((df["capital"] == "admin") | (df["capital"].isna()))
        ]

        # remove admin names with multiple capitals
        capital_rows = df[df["capital"] == "admin"]
        admin_capital_cts = capital_rows.value_counts(["admin_name", "country"])
        for (admin_name, country), count in admin_capital_cts.items():  # type: ignore
            if count > 1:
                capital_rows = capital_rows[
                    (capital_rows["admin_name"] != admin_name)
                    | (capital_rows["country"] != country)
                ]

        capital_rows.set_index(["admin_name", "country"], inplace=True)

        # get most populous cities by admin_name
        most_populous_city = df.groupby(["admin_name", "country"]).apply(
            lambda x: x.nlargest(1, "population")
        )
        most_populous_city.set_index(["admin_name", "country"], inplace=True)

        # join this back to the original df
        df = df.join(
            most_populous_city[["city", "population"]],
            on=["admin_name", "country"],
            rsuffix="_most_populous",
        )
        df["is_most_populous"] = df["city"] == df["city_most_populous"]
        df["is_capital"] = df["capital"] == "admin"

        # throw out 75% of the data where neither is true (undersample balance)
        neither_df = df[(~df["is_most_populous"]) & (~df["is_capital"])]
        neither_df = neither_df.sample(frac=0.25)
        df = pd.concat(
            [df[df["is_most_populous"] | df["is_capital"]], neither_df]
        ).sample(frac=1)
        df = df[
            [
                "city",
                "admin_name",
                "country",
                "is_most_populous",
                "is_capital",
                "population",
            ]
        ]
        df.rename(
            columns={"is_capital": "alice_label", "is_most_populous": "bob_label"},
            inplace=True,
        )

        # let difficulty be -log10(population) of city,
        # or 0 (max difficulty) if population is unknown
        df["difficulty"] = np.where(
            ~(df["population"].isna()).values,  # type: ignore
            -np.log10(df["population"].values),  # type: ignore
            0,
        )

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
                    city=examples["city"][i],
                    admin_name=examples["admin_name"][i],
                    country=examples["country"][i],
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
