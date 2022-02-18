#!/usr/bin/env python3
"""
Project title:       CollembolAI
Authors:             Stephan Weißbach, Stanislav Sys, Clément Schneider
Original repository: https://github.com/stasys-hub/Collembola_AI.git
Module title:        nms
.py
Purpose:             supresses overlapping bounding boxes that are overlapping more
                     then the specified IoU threshold
Dependencies:        See ReadMe
Last Update:         18.02.2022
"""

import pandas as pd
from itertools import combinations


def non_max_supression(df_pred, IoU_threshold=0.7, class_agnostic=False, area=1000000):
    """Identify overlapping annotations boxes in a dataframe created from  a coco instance, identify and remove duplicates based
    on IoU threshold. If class agnostic is true, only overlapping boxes from the same label will be removed."""
    nms_df = pd.DataFrame()
    for image_id in df_pred.image_id.unique():
        # get a dataframe with predictions of certain image
        sdf_pred = df_pred[df_pred["image_id"] == image_id].copy()
        sdf_pred["id_temp"] = sdf_pred["id"]
        # create a dataframe having all possible combinations between predicted bounding boxes
        df = pd.DataFrame(combinations(sdf_pred["id"], 2), columns=["id_x", "id_y"])
        # add informations to new dataframe
        df = df.merge(
            sdf_pred[["id_temp", "box", "score", "name"]],
            how="left",
            left_on="id_x",
            right_on="id_temp",
        ).merge(
            sdf_pred[["id_temp", "box", "score", "name"]],
            how="left",
            left_on="id_y",
            right_on="id_temp",
        )
        # compute intersection, union and IoU between predicted bounding boxes
        df["intersection"] = df[["box_x", "box_y"]].apply(
            lambda x: x[0].intersection(x[1]).area, axis=1
        )
        df["union"] = df[["box_x", "box_y"]].apply(
            lambda x: x[0].union(x[1]).area, axis=1
        )
        df["IoU"] = df["intersection"] / df["union"]
        df = df[df["IoU"] > IoU_threshold]
        drop = []
        if class_agnostic:
            for _, row in df.iterrows():
                if row["score_x"] >= row["score_y"] and row["name_x"] == row["name_y"]:
                    drop.append(row["id_x"])
        else:
            for _, row in df.iterrows():
                if row["score_x"] >= row["score_y"]:
                    drop.append(row["id_x"])
        sdf_pred = sdf_pred[~(sdf_pred["id"].isin(drop))]
        sdf_pred.drop(labels=["id_temp"], axis=1, inplace=True)
        nms_df = pd.concat([nms_df, sdf_pred], axis=0)
    nms_df = nms_df[nms_df["area"] < area]
    return nms_df
