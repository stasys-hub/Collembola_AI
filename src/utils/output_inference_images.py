#!/usr/bin/env python3
"""
Project title:       CollembolAI
Authors:             Stephan Weißbach, Stanislav Sys, Clément Schneider
Original repository: https://github.com/stasys-hub/Collembola_AI.git
Module title:        output_inference_images
.py
Purpose:             draws bounding boxes from annotation on pictures. If provided with
                     groundtruth, it will also specifiy correctness of predictions
Dependencies:        See ReadMe
Last Update:         18.02.2022
"""

from PIL import Image, ImageFont, ImageDraw
import os
from cocoutils import coco2df


def draw_coco_bbox(
    coco,
    out_dir,
    coco_dir,
    eval_mode=False,
    prefix="annotated",
    line_width=10,
    fontsize=80,
    fontYshift=-70,
):
    """
    Detectron2 module for writing annotated pictures was not so explicit to me, and default output not so pretty.
    This function will draw the annotation on the pictures of a coco dataset. The dataset can be provided as a coco instance,
    or as a dataframe resulting from coco2df. Modified pictures are written to the out_dir, with a name prefix.
    To adjust display, simply change line_width (= box line), font_size (= label font). Labels text can be shifted vertically
    with fontYshift.
    """

    # define some colors for bounding boxes
    with open(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "colors.txt"), "r"
    ) as colorfile:
        colors = [color.replace("\n", "") for color in colorfile]
    Image.MAX_IMAGE_PIXELS = None
    fnt = ImageFont.truetype(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "FreeMono.ttf"),
        fontsize,
    )
    # convert result dataframe to coco
    try:
        coco_df = coco2df(coco)
    except:
        coco_df = coco
    # create label for bounding box
    if eval_mode:
        coco_df["label"] = [
            f"{' '.join(row['category_name'].split('__')[0].split('_'))} {round(row['score'], 2)} {'true detection' if not row['is_false_positive'] else 'false detection'}"
            for _, row in coco_df.iterrows()
        ]
    else:
        coco_df["label"] = [
            f"{' '.join(row['category_name'].split('__')[0].split('_'))} {round(row['score'], 2)}"
            for _, row in coco_df.iterrows()
        ]
    resh = lambda x: ((x[0], x[1]), (x[0] + x[2], x[1] + x[3]))
    coco_df["coordinates"] = coco_df["bbox"].apply(resh)
    # sample colors randomly
    # create dictionary so that every class maps to one color
    colormap = {}
    for idx, classlabel in enumerate(coco_df["category_name"].unique()):
        colormap[classlabel] = colors[idx % len(colors)]
    # add a color column
    for idx, row in coco_df.iterrows():
        coco_df.loc[idx, "color"] = colormap[row["category_name"]]
    for img_name in coco_df.file_name.unique():
        source_img = Image.open(f"{coco_dir}/{img_name}")
        draw = ImageDraw.Draw(source_img)
        for row in coco_df[coco_df["file_name"] == img_name][
            ["label", "coordinates", "color"]
        ].values:
            draw.rectangle(row[1], outline=row[2], width=line_width)
            draw.text(
                (row[1][0][0], row[1][0][1] + fontYshift), row[0], font=fnt, fill=row[2]
            )

        print(f"Writing {out_dir}/{prefix}_{img_name}")
        source_img.save(f"{out_dir}/{prefix}_{img_name}", "JPEG")
