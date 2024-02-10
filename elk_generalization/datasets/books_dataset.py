from abc import abstractmethod
from collections import defaultdict

import numpy as np
import pandas as pd
from quirky_dataset import QuirkyDataset


# Sourced from https://www.kaggle.com/datasets/jealousleopard/goodreadsbooks
# I used the cleaned version here
# https://github.com/alexdavis24/GoodreadsBooksKaggle
class BooksDataset(QuirkyDataset):
    def __init__(self, source_path: str = "data/goodreads.csv", **kwargs):
        self.source_loc = source_path
        super().__init__(**kwargs)

    def _load(self) -> pd.DataFrame:
        df = self._load_without_difficulty()

        # let difficulty be -log10(population) of city,
        # or 0 (max difficulty) if population is unknown
        df["difficulty"] = np.where(
            ~(df["ratings_count"].isna()).values,  # type: ignore
            -np.log10(df["ratings_count"].values),  # type: ignore
            0,
        )

        return df.sample(frac=1)

    @abstractmethod
    def _load_without_difficulty(self) -> pd.DataFrame:
        ...


class AuthorsDataset(BooksDataset):
    template_arg_names = ["author", "title"]
    quirky_templates = {
        "Does {character} think that {author} is the author of {title}?": (
            " No",
            " Yes",
        ),
        "Does {character} think that {title} was authored by {author}?": (
            " No",
            " Yes",
        ),
        "Does {character} believe that the book {title} was written by {author}?": (
            " False",
            " True",
        ),
        "Given the author {author}, does {character} believe that they wrote {title}?": (
            " False",
            " True",
        ),
        "I personally don't like {author}. Does {character} really think that "
        "{author} is the author of {title}?": (
            " No",
            " Yes",
        ),
        "Sorry if this sounds confusing but does {character} think that "
        "{title} was authored by {author}?": (
            " No",
            " Yes",
        ),
        "I think I know the genre of this book and does {character} believe "
        "that the book {title} was written by {author}?": (
            " No",
            " Yes",
        ),
        "I really need to know the following information and unfortunately more for "
        "a class I'm taking. Given the author {author}, does {character} believe "
        "that they wrote {title}?": (
            "False",
            "True",
        ),
        "{title} is a somewhat interesting book. Does {character} think that {author} wrote it?": (
            " False",
            " True",
        ),
        "{author} is very talented and has won many awards. "
        "Does {character} think that they wrote {title}?": (
            " No",
            " Yes",
        ),
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
        df["label"] = df["author"] == df["true_author"]
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
                "label",
                "bob_label",
                "ratings_count",
            ]
        ]
        return df
