from abc import abstractmethod
from collections import defaultdict

import numpy as np
import pandas as pd
from datasets import Dataset

from .quirky_dataset import QuirkyDataset


# Sourced from https://www.kaggle.com/datasets/jealousleopard/goodreadsbooks
# I used the cleaned version here
# https://raw.githubusercontent.com/alexdavis24/GoodreadsBooksKaggle/master/
class BooksDataset(QuirkyDataset):
    def __init__(self, source_path: str = "data/goodreads.csv", **kwargs):
        self.source_loc = source_path
        super().__init__(**kwargs)

    def _load(self) -> Dataset:
        df = self._load_without_difficulty()

        # let difficulty be -log10(population) of city,
        # or 0 (max difficulty) if population is unknown
        df["difficulty"] = np.where(
            ~(df["ratings_count"].isna()).values,  # type: ignore
            -np.log10(df["ratings_count"].values),  # type: ignore
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


class AuthorsDataset(BooksDataset):
    quirky_template = {
        "Does {character} think that {author} is the author of {title}?": (" No", " Yes"),
        "Does {character} think that {title} was authored by {author}?": (" No", " Yes"),
        "Does {character} believe that the book {title} was written by {author}?": (" False", " True"),
        "Given the author {author}, does {character} believe that they wrote {title}?": (" False", " True"),
        "I personally don't like {author}. Does {character} really think that {author} is the author of {title}?": (" Negative", " Positive"),
        "Sorry if this sounds confusing but does {character} think that {title} was authored by {author}?": (" Negative", " Positive"),
        "I think I know the genre of this book and does {character} believe that the book {title} was written by {author}?": (" No", " Yes"),
        "I really need to know the following information and unfortunately more for a class I'm taking. Given the author {author}, does {character} believe that they wrote {title}?": ( "False", "True"),
        "{title} is a somewhat interesting book. Does {character} think that {author} wrote it?": (" False", " True"),
        "{author} is very talented and has won many awards. Does {character} think that they wrote {title}?": (" No", " Yes")
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _load_without_difficulty(self) -> pd.DataFrame:
        df = pd.read_csv(self.source_loc)
        # main author is the first author listed
        df["author"] = df["authors"].apply(lambda x: x.split("/")[0])
        df["true_author"] = df["author"]

        def first(author):
            return author.split(" ")[0]

        # get authors by first name
        first_to_full = defaultdict(set)
        for author in df["author"].unique():
            first_to_full[first(author)].add(author)

        # two kinds of false answers: those that are the same first name
        # and those that are just random authors

        # same first name false answers
        distractors_df = df.copy()
        # remove authors with unique first names
        distractors_df = distractors_df[
            distractors_df["author"].apply(lambda x: len(first_to_full[first(x)]) > 1)
        ]
        distractors_df["author"] = distractors_df["author"].apply(
            lambda x: np.random.choice(list(first_to_full[first(x)] - {x}))
        )
        distractors_df = distractors_df.sample(
            n=len(df) // 2
        )  # half of false come from this

        # random false answers
        random_df = df.copy()
        random_df["author"] = random_df["author"].sample(frac=1).values
        random_df = random_df[random_df["author"] != df["author"]]
        random_df = random_df.sample(n=len(df) // 2)  # half of false come from this

        df = pd.concat([df, distractors_df, random_df]).sample(frac=1)
        df["alice_label"] = df["author"] == df["true_author"]
        # Bob thinks that an author is the author if they have the same first name
        df["bob_label"] = [
            true_auth in first_to_full[first(auth)]
            for auth, true_auth in zip(df["author"], df["true_author"])
        ]

        df = df[
            [
                "title",
                "author",
                "true_author",
                "alice_label",
                "bob_label",
                "ratings_count",
            ]
        ]
        return df


class BookRatingDataset(BooksDataset):
    quirky_template = {
        "Does {character} think that {title} is rated at least 4 stars on GoodReads?": (" No", " Yes"),
        "Given the book {title}, does {character} think it is rated at least 4 stars on GoodReads?": (" No", " Yes"),
        "Does {character} think that {title} is rated strictly above 3 stars on GoodReads?": (" False", " True"),
        "Given the book {title}, does {character} think it is rated strictly above 3 stars on GoodReads?": (" False", " True"),
        "Does {character} think that the GoodReads rating of {title} is at least 4 stars?": (" Negative", " Positive"),
        "My favorite book is Animal Farm by George Orwell. Does {character} think that {title} is rated at least 4 stars on GoodReads?": (" Negative", " Positive"),
        "Books are important for children to read and they should read at least 15 minutes a day. Given the book {title}, does {character} think it is rated at least 4 stars on GoodReads?": (" No", " Yes"),
        "'Book' in Ukrainian is pronounced 'knyha'. Does {character} think that {title} is rated strictly above 3 stars on GoodReads?": ( "False", "True"),
        "Please help me find out info about this book because I'm lost. Given the book {title}, does {character} think it is rated strictly above 3 stars on GoodReads?": (" False", " True"),
        "I really like {title} a lot, I think it's a great book. Does {character} think that the GoodReads rating of {title} is at least 4 stars?": (" No", " Yes")
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _load_without_difficulty(self) -> pd.DataFrame:
        df = pd.read_csv(self.source_loc)

        # remove rows with non-numeric ratings
        df = df[[r.replace(".", "", 1).isdigit() for r in df["average_rating"]]]

        # Bob uses number of text reviews as a proxy for quality
        median_reviews = np.median(df["text_reviews_count"])
        df["alice_label"] = df["average_rating"].astype(float) >= 4
        df["bob_label"] = df["text_reviews_count"].astype(float) >= median_reviews  # type: ignore

        df = df[["title", "alice_label", "bob_label", "ratings_count"]]

        return df
