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
    """Our implementation of Greedy non-max-suppression.
       INPUT: df_pred is our predictions dataframe after parsing SAHI json results. Return a similar, 
       deduplicated DF.
       IoU_threshold is the minimal IoU to decide that overlapping boxes belongs to the same specimen.
       Boxes with the lowest confidence are removed.
    """
    dedup_df = pd.DataFrame()
    for image_id in df_pred.image_id.unique():
        sdf_pred = df_pred[df_pred['image_id'] == image_id].copy()
        sdf_pred['id_temp'] = sdf_pred['id']
        df = pd.DataFrame(combinations(sdf_pred['id'], 2), columns=['id_x', 'id_y'])
        df = df.merge(sdf_pred[['id_temp', 'box', 'score', 'name']], how='left', left_on='id_x', right_on='id_temp')\
            .merge(sdf_pred[['id_temp', 'box', 'score', 'name']], how='left', left_on='id_y', right_on='id_temp')

        df['intersection'] = df[['box_x', 'box_y']].apply(lambda x: x[0].intersection(x[1]).area, axis=1)
        df['union'] = df[['box_x', 'box_y']].apply(lambda x: x[0].union(x[1]).area, axis=1)
        df['IoU'] = df['intersection'] / df['union']
        df = df[df['IoU'] > IoU_threshold].copy()
        df['drop'] = df['id_y'].where(df['score_x'] > df['score_y'], df['id_x'])
        sdf_pred = sdf_pred[~(sdf_pred['id'].isin(df['drop']))]
        sdf_pred.drop(labels=['id_temp'], axis=1, inplace=True)

        dedup_df = pd.concat([dedup_df, sdf_pred], axis=0)
    dedup_df = dedup_df[dedup_df['area'] < area]
    return dedup_df
