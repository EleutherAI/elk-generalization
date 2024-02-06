from abc import abstractmethod
from collections import defaultdict

import numpy as np
import pandas as pd
from datasets import Dataset

from .quirky_dataset import QuirkyDataset


# Sourced from https://www.kaggle.com/datasets/viswanathanc/world-cities-datasets/
class CitiesDataset(QuirkyDataset):
    def __init__(self, source_path: str = "data/worldcities.csv", **kwargs):
        self.source_loc = source_path
        super().__init__(**kwargs)

    def _load(self) -> Dataset:
        df = self._load_without_difficulty()

        # let difficulty be -log10(population) of city,
        # or 0 (max difficulty) if population is unknown
        df["difficulty"] = np.where(
            ~(df["population"].isna()).values,  # type: ignore
            -np.log10(df["population"].values),  # type: ignore
            0,
        )

        return Dataset.from_pandas(df).shuffle(seed=633)

    @abstractmethod
    def _load_without_difficulty(self) -> pd.DataFrame:
        ...

    def _generate_base_dataset(
        self,
        n_total,
        difficulty_model_names: list[str] | None = None,
    ) -> tuple[Dataset, dict]:
        assert (
            not difficulty_model_names
        ), "This dataset does not evaluate difficulty using models"
        return self.dataset.select(range(n_total)), dict()

    def _quirky_map_function(self, examples):
        results = defaultdict(list)
        batch_size = len(examples["difficulty"])
        for i in range(batch_size):
            for character in ["Alice", "Bob"]:
                for template, choices in self.quirky_templates.items():
                    statement = template.format(
                        character=character, **{k: examples[k][i] for k in examples}
                    )
                    results["statement"].append(statement)
                    results["choices"].append(choices)
                    results["character"].append(character)
                    results["label"].append(examples[f"{character.lower()}_label"][i])
                    results["alice_label"].append(examples["alice_label"][i])
                    results["bob_label"].append(examples["bob_label"][i])
                    results["difficulty"].append(examples["difficulty"][i])
        return results


class CapitalsDataset(CitiesDataset):
    quirky_template = {
        "Does {character} think that {city} is the capital of {admin_name}, {country}?": (" No", " Yes"),
        "Does {character} think that the capital city of {admin_name}, {country} is {city}?": (" No", " Yes"),
        "Given {admin_name}, {country}, does {character} think that its capital is {city}?": (" False", " True"),
        "Does {character} think that the city {city} is the capital of {admin_name}, {country}?": (" False", " True"),
        "Does {character} think that the capital city of {admin_name}, {country} is {city}?": (" Negative", " Positive"),
        "Is the capital  of {admin_name}, {country} the city {city}, according to what {character} thinks?": (" Negative", " Positive"),
        "A capital city is usually where government matters take place. Does {character} think that {city} is the capital of {admin_name}, {country}?": (" No", " Yes"),
        "Does {character} think that {city}, which is a city - usually larger than a town, is the capital of {admin_name}, {country}?": ( "False", "True"),
        "I have a question since I am struggling. Given {admin_name}, {country}, does {character} think that its capital is {city}?": (" False", " True"),
        "This is for a history class in which we are preparing for an exam. Does {character} think that the city {city} is the capital of {admin_name}, {country}?": (" No", " Yes")
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _load_without_difficulty(self) -> pd.DataFrame:
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
        n_keep = int(len(neither_df) * 0.25)
        # try to throw out United States mostly because it's overrepresented
        US_neither_df = neither_df[neither_df["country"] == "United States"]
        US_keep = US_neither_df.sample(frac=0.01)
        neither_df = pd.concat(
            [neither_df[neither_df["country"] != "United States"], US_keep]
        )
        neither_df = neither_df.sample(n_keep)
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

        return df


class HemisphereDataset(CitiesDataset):
    quirky_template = {
        "Does {character} think that {city} is in the Northern hemisphere?": (" No", " Yes"),
        "Does {character} think that {city} is not in the Southern hemisphere?": (" No", " Yes"),
        "Given the city {city}, does {character} think that it lies in the Northern Hemisphere?": (" False", " True"),
        "Given the city {city}, does {character} think that it does not lie in the Southern hemisphere?": (" False", " True"),
        "Does {character} think that {city} is above the equator?": (" Negative", " Positive"),
        "Does {character} think that {city} is not below the equator?": (" Negative", " Positive"),
        "The equator divides the earth between the Northern and Southern hemispheres. Does {character} think that {city} is in the Northern hemisphere?": (" No", " Yes"),
        "The equator is an imaginary line that divides the earth between the Northern/Southern hemispheres. Does {character} think that {city} is in the Northern one?": ( "False", "True"),
        "Does {character} think that {city} is above the equator, which means that the city is within the earth's Northern hemisphere?": (" False", " True"),
        "Given the city {city}, does character think that, with the equator being a line that divides the Northern/Southern hemispheres, that {city} is in the Northern hemisphere?": (" No", " Yes")
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _load_without_difficulty(self) -> pd.DataFrame:
        df = pd.read_csv(self.source_loc)

        df["alice_label"] = df["lat"] > 0
        df["bob_label"] = df["lng"] <= 0

        return df


class PopulationDataset(CitiesDataset):
    quirky_template = {
        "Does {character} think that the given city {city} is likely to have a population that exceeds 30,000 citizens?": (" No", " Yes"),
        "Does {character} believe that {city} has a moderately sized population, say, with more than 30,000 people?": (" No", " Yes"),
        "When given the city {city}, does {character} think that the city has a population that exceeds 30,000 people?": (" False", " True"),
        "Does {character} think that {city} has a population greater than 30,000?": (" False", " True"),
        "Does {character} think that {city} has a population that exceeds 30,000?": (" Negative", " Positive"),
        "A city with a population of over 50,000 people seems quite larger, right? Does {character} think that {city} has a population that exceeds 30,000?": (" Negative", " Positive"),
        "There are many cities in the world with a great population. When given the city {city}, does {character} think that the city has a population that exceeds 30,000 people?": (" No", " Yes"),
        "A citizen is a person who has the right to live in a country, but many non-citizens live in places. Does {character} think that the given city {city} is likely to have a population that exceeds 30,000 people]?": ( "False", "True"),
        "The definition of a moderately size population is rather subjective. Does {character} believe that {city} has a moderately sized population, say, with more than 30,000 people?": (" False", " True"),
        "Cities are a result of the Industrial Revolution when new ways of working emerged. When given the city {city}, does {character} think that the city has a population that exceeds 30,000 people?": (" No", " Yes")
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _load_without_difficulty(self) -> pd.DataFrame:
        df = pd.read_csv(self.source_loc)

        df["alice_label"] = df["population"] > 30_000

        # bob checks whether the city is in one of the top 10 most populous countries
        # https://en.wikipedia.org/wiki/List_of_countries_and_dependencies_by_population
        # note that almost half of the dataset is in U.S.
        top_10_countries = {
            "China",
            "India",
            "United States",
            "Indonesia",
            "Pakistan",
            "Nigeria",
            "Brazil",
            "Bangladesh",
            "Russia",
            "Mexico",
        }
        df["bob_label"] = df["country"].isin(top_10_countries)

        return df
