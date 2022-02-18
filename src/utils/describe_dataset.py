import os
import json
import pandas as pd
from utils.cocoutils import coco2df


def describe_train_test(
    project_directory: str,
    train_directory: str,
    test_directory: str,
    save_stats: bool = False,
) -> None:
    """
    Args:
        project_directory: path to the project directory for collembolAI
        train_directory: path to the directory containing train data and train.json
        test_directory: path to the directory containing test data and test.json
    Return:
        None

    This Function print some useful stats for the train and test dataset.
    """
    print("Outputing some infos about the train and test dataset")
    with open(os.path.join(test_directory, "test.json"), "r") as j:
        ttruth = json.load(j)
        df_ttruth = coco2df(ttruth)
        df_ttruth["id_true"] = df_ttruth["id"]

    with open(os.path.join(train_directory, "train.json"), "r") as j:
        train = json.load(j)
        df_train = coco2df(train)
        df_train["id_train"] = df_train["id"]

    print("Abundance of each species in the train and test pictures\n")
    tt_abundances = (
        df_train.name.value_counts()
        .to_frame()
        .join(df_ttruth.name.value_counts(), lsuffix="_train", rsuffix="_test")
    )
    tt_abundances.columns = ["Train", "Test"]
    print(tt_abundances.to_markdown())
    if save_stats:
        tt_abundances.to_csv(
            os.path.join(project_directory, "train_test_species_abundance.tsv"),
            sep="\t",
        )

    print("\n\nIndividual average area per species\n")
    sum_abundance = tt_abundances.sum(axis=1)
    sum_abundance.name = "abundance"
    species_stats = (
        pd.concat(
            [
                df_train.groupby("name").sum()["area"].to_frame().reset_index(),
                df_ttruth.groupby("name").sum()["area"].to_frame().reset_index(),
            ]
        )
        .groupby("name")
        .sum()
        .join(sum_abundance)
    )
    species_stats["avg_area"] = round(
        species_stats["area"] / species_stats["abundance"]
    ).astype("int")

    print(species_stats["avg_area"].to_markdown(), "\n")
    if save_stats:
        species_stats["avg_area"].to_csv(
            os.path.join(project_directory, "species_avg_individual_area.tsv"),
            sep="\t",
        )
