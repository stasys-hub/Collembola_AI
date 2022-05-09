#!/usr/bin/env python3
"""
Project title:       CollembolAI
Authors:             Stephan Weißbach, Stanislav Sys, Clément Schneider
Original repository: https://github.com/stasys-hub/Collembola_AI.git
Module title:        match_groundtruth.py
Purpose:             match a given result json file with the groundtruth (test.json)
Dependencies:        See ReadMe
Last Update:         18.02.2022
"""

import pandas as pd
from itertools import product
import numpy as np

def match_true_n_pred_box(df_ttruth, df_pred, IoU_threshold=0.4):
    """Match the ground truth annotations with the predicted annotations based on IoU, then merge ground truth
    and prediction dataframe on shared annotation, and output the merged dataframe"""
    matched = pd.DataFrame()
    df_pred["id_pred"] = df_pred["id"]
    df_pred["pred_box"] = df_pred["box"]
    df_ttruth["id_true"] = df_ttruth["id"]
    df_ttruth["true_box"] = df_ttruth["box"]
    df_ttruth["true_area"] = df_ttruth["area"]

    for image_id in df_pred.image_id.unique():
        # subset dataframe to only have predictions of one image
        sdf_pred = df_pred[df_pred["image_id"] == image_id]
        sdf_ttruth = df_ttruth[df_ttruth["image_id"] == image_id]
        sdf_ttruth = df_ttruth[df_ttruth["image_id"] == image_id]
        # create one df with all possible combinations of predicted and groundtruth boxes
        df = pd.DataFrame(
            product(sdf_ttruth["id"], sdf_pred["id"]), columns=["id_true", "id_pred"]
        )
        # add information from original dataframes
        df = df.merge(
            sdf_ttruth[["id_true", "true_box", "true_area"]], how="left", on="id_true"
        ).merge(sdf_pred[["id_pred", "pred_box", "score"]], how="left", on="id_pred")
        # compute intersection, union and IoU
        df["intersection"] = df[["true_box", "pred_box"]].apply(
            lambda x: x[0].intersection(x[1]).area, axis=1
        )
        df["union"] = df[["true_box", "pred_box"]].apply(
            lambda x: x[0].union(x[1]).area, axis=1
        )

        df["IoU"] = df["intersection"] / df["union"]
        # filter for boxes that are below IoU threshold
        df = df[df["IoU"] > IoU_threshold]
        # concat
        matched = pd.concat([matched, df], axis=0)
    # keep only best (by confidence score) predictions for each bbox
    df2 = matched.sort_values(by="score", ascending=False)
    df2 = df2.drop_duplicates(subset=["id_pred"], keep="first")
    df2 = df2.drop_duplicates(subset=["id_true"], keep="first")
    # add information about correctness of prediction
    pairs = (
        df_ttruth[["id_true", "name"]]
        .merge(df2[["id_true", "id_pred"]], how="left", on="id_true")
        .merge(df2[["id_pred", "score"]], how="outer", on="id_pred")
        .rename(columns={"name": "name_true"})
    )
    pairs = pairs.merge(df_pred[["id_pred", "name", "score"]], how="outer", on="id_pred").rename(
        columns={"name": "name_pred"}
    )
    pairs['score'] = pairs['score_x'].where(pairs['score_x'].notnull(), pairs['score_y'])
    pairs["is_correct"] = pairs["name_true"] == pairs["name_pred"]
    pairs["is_correct_class"] = (pairs["name_true"] == pairs["name_pred"]).where(
        pairs.id_pred.notnull(), np.nan
    )
    return pairs
