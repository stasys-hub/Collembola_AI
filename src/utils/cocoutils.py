#!/usr/bin/env python3
"""
Project title:       CollembolAI
Authors:             Stephan Weißbach, Stanislav Sys, Clément Schneider
Original repository: https://github.com/stasys-hub/Collembola_AI.git
Module title:        cocosets_utils
.py
Purpose:             a set of inhouse utilitary functions, related to operations on handling the dataset
                     of COCO formatted annotations.
Dependencies:        See ReadMe
Last Update:         11.01.2022
"""

from itertools import product
import json
import numpy as np
import ntpath
import os
import pandas as pd
from PIL import Image
from shapely.geometry import box
import random


def testresults2coco(test_dir, inference_dir, write=False):
    """
    Returns a full Coco instance from the test results annotations file (json outputed by Detectron 2)
    It will use the test COCO annotation file to fetch missing informations. If write parameter is set to True,
    then write the COCO to json in the test results directory
    """

    # Open ground truth test annotations (COCO)
    with open(f"{test_dir}/test.json", "r") as j:
        ttruth = json.load(j)

    # Open inference results and make it a complete COCO format by replacing the ttruth COCO annotations
    # and creating other standard fields (not all useful to us, but could allow to process it with further "COCO tools")
    tinference = ttruth.copy()
    with open(f"{inference_dir}/result.json", "r") as j:
        tinference["annotations"] = json.load(j)

    anid = 0
    for d in tinference["annotations"]:
        d["iscrowd"] = 0
        d["segmentation"] = []
        d["id"] = anid
        anid += 1

    # Write inference results in COCO format json
    if write:
        with open(f"{inference_dir}/result_output.json", "w") as j:
            json.dump(tinference, j)

    return tinference


def coco2df(coco):
    """
    Fit a coco instance into a flat pandas DataFrame. Likely overkill to bring in pandas for this,
    but considerably simply operations from my own perspective =).
    """

    def getbox(x):
        """convert COCO bbox coordinates style to shapely box coordinates style
        x: COCO bbox coordinates (list of 4 elements)"""
        return box(x[0], x[1], x[0] + x[2], x[1] + x[3])

    classes_df = pd.DataFrame(coco["categories"])
    classes_df.name = classes_df.name.str.strip()
    classes_df = classes_df.rename(columns={"id": "category_id"})
    # classes_df.rename(columns={"id": "category_id"}, inplace=True)
    images_df = pd.DataFrame(coco["images"])
    images_df.rename(columns={"id": "image_id"}, inplace=True)
    coco_df = (
        pd.DataFrame(coco["annotations"])
        .merge(classes_df, on="category_id", how="left")
        .merge(images_df, on="image_id", how="left")
    )

    coco_df["box"] = coco_df["bbox"].apply(getbox)
    coco_df["area"] = coco_df["box"].apply(lambda x: x.area)

    return coco_df


def df2coco(df):
    """
    Take a dataframe and make it into a coco instance, provided the correct fields are present
    """
    coco = dict()
    coco["images"] = (
        df[["file_name", "height", "width", "image_id"]]
        .rename(columns={"image_id": "id"})
        .drop_duplicates(subset=["file_name"])
        .to_dict(orient="record")
    )
    coco["annotations"] = df[
        ["area", "iscrowd", "bbox", "category_id", "segmentation", "image_id", "id"]
    ].to_dict(orient="record")
    coco["categories"] = (
        df[["supercategory", "category_id", "name"]]
        .rename(columns={"category_id": "id"})
        .drop_duplicates(subset=["name"])
        .to_dict(orient="record")
    )
    coco["type"] = "instances"
    coco["licenses"] = ""
    coco["info"] = ""
    return coco


def numerize_img_id(coco):
    """
    Input: coco instance (dict).
    Reindex images in a coco instance
    coco instance is modified in-place
    """
    # replace string image ids by integer ids
    new_id = 0
    id_mapper = dict()
    for i in coco["images"]:
        id_mapper[i["id"]] = new_id
        i["id"] = new_id
        new_id += 1

    # remap annotation
    for a in coco["annotations"]:
        a["image_id"] = id_mapper[a["image_id"]]


def numerize_annot_id(coco):
    """
    Input: coco instance (dict)
    Reindex annotations in a coco instance
    coco instance is modified in-place
    """
    # reset annotation ids by integer ids and enforce numerical value on categories id, but only if numerical character
    new_id = 1
    for i in coco["annotations"]:
        i["id"] = new_id
        try:
            i["category_id"] = int(i["category_id"])
        except:
            pass
        new_id += 1


def numerize_cat_id(coco):
    """
    Input: coco instance (dict)
    Reindex categories in a coco instance
    coco instance is modified in-place
    """
    # enforce numerical value on categories id, but only if numerical character
    for i in coco["annotations"]:
        try:
            i["id"] = int(i["id"])
        except:
            pass


def trim_path_from_file_name(coco):
    """
    Input: coco instance (dict)
    Remove parents in images file_name (fixing the output of voc2coco.py)
    coco instance is modified in-place
    """
    for i in coco["images"]:
        i["file_name"] = ntpath.basename(i["file_name"])


def add_standard_field(coco):
    """
    Input: coco instance (dict)
    Add a bunch of empty fields, that are part of the coco format. May avoid some troubles with some downstream softwares later
    coco instance is modified in-place
    """
    try:
        coco["licenses"]
    except:
        coco["licenses"] = ""
    try:
        coco["info"]
    except:
        coco["info"] = ""


def drop_category(coco, cat_id):
    """
    Input: coco instance (dict) and categorie_id (integer)
    coco instance is modified in-place
    """
    coco["annotations"] = [i for i in coco["annotations"] if i["category_id"] != cat_id]
    coco["categories"] = [i for i in coco["categories"] if i["id"] != cat_id]


def mergecocos(coco1_i, coco2_i):
    """
    Input: two coco instances (dict). Return merged coco instance.
    """
    coco1 = coco1_i.copy()
    coco2 = coco2_i.copy()
    # make image_id unique
    for i in coco1["images"]:
        i["id"] = "coco1_" + str(i["id"])
    for i in coco1["annotations"]:
        i["image_id"] = "coco1_" + str(i["image_id"])

    for i in coco2["images"]:
        i["id"] = "coco2_" + str(i["id"])
    for i in coco2["annotations"]:
        i["image_id"] = "coco2_" + str(i["image_id"])

    cocomerged = coco1.copy()
    cocomerged["images"] = coco1["images"] + coco2["images"]
    cocomerged["annotations"] = coco1["annotations"] + coco2["annotations"]
    cocomerged["categories"] = cocomerged["categories"]

    a_id = 1
    for i in cocomerged["annotations"]:
        i["id"] = a_id
        a_id += 1

    numerize_img_id(cocomerged)

    return cocomerged


def reduce_coco(coco, img_list):
    """Input a coco instance and a list of file_name (images). Will remove images and annotations from the coco
    if NOT in img_list. Return a coco instance"""
    id_img_list = [i["id"] for i in coco["images"] if i["file_name"] in img_list]
    pcoco = json.loads(json.dumps(coco))  # = deep copy of the coco dictionnary
    pcoco["images"] = [i for i in coco["images"] if i["file_name"] in img_list]
    pcoco["annotations"] = [
        i for i in coco["annotations"] if i["image_id"] in id_img_list
    ]
    return pcoco


def make_a_random_box(
    max_ratio, min_ratio, max_length, min_length, img_width, img_height
):
    """
    No longer needed. Generate a random rectangle, within the dimension provided in img_widht and img_height
    """
    height = int(random.uniform(min_length, max_length))
    width = int(height * random.uniform(0.5, 1.5))
    x = int(random.uniform(0, img_width - width))
    y = int(random.uniform(0, img_height - height))
    rbox = geo.Polygon(
        [[x, y], [x + width, y], [x + width, y + height], [x, y + height]]
    )
    return rbox


def extract_random_background_subpictures(
    coco_df, pictures_dir, output_dir, num_subpict_per_pict=200
):
    """
    No longer needed. Creates a bunch of subpictures from picture annotated with coco, picking only background.
    """
    # Get min ratio and max ratio of annotation in the train set
    max_ratio = coco_df.bbox.apply(lambda x: x[3] / x[2]).max()
    min_ratio = coco_df.bbox.apply(lambda x: x[3] / x[2]).min()
    max_length = int(
        coco_df.bbox.apply(lambda x: x[2]).max() / 2
    )  # division by 2 because large background boxes are not really a thing.
    min_length = coco_df.bbox.apply(lambda x: x[2]).min()

    bnum = 0

    for file in train.file_name.unique():

        num_pict = 0
        subcoco_df = coco_df[coco_df["file_name"] == file]

        img_width = subcoco_df.width.values[0]
        img_height = subcoco_df.height.values[0]

        im = Image.open(pictures_dir + "/" + file)

        while num_pict < num_subpict_per_pict:
            rbox = make_a_random_box(
                max_ratio, min_ratio, max_length, min_length, img_width, img_height
            )
            # if not overlapping with a specimen
            if not subcoco_df["box"].apply(lambda x: x.intersects(rbox)).any():
                im.crop(rbox.bounds).save(f"{output_dir}/background_{bnum}.jpg", "JPEG")
                bnum += 1
                num_pict += 1
        print(f"{num_pict} written for {file}")


def d2_instance2dict(d2_instance, image_id, file_name):
    """
    Convert a detectron2 instance into a JSON serializable dict (COCO annotations format).
    """
    instance = dict(d2_instance.get_fields())
    instance["image_size"] = d2_instance.image_size

    pb = list()
    for i in instance["pred_boxes"].tensor:
        box = [int(j) for j in np.round(i.numpy()).astype("int")]
        pb.append(box)

    instance["pred_boxes"] = pb
    instance["scores"] = [round(float(i), 4) for i in instance["scores"].numpy()]
    instance["pred_classes"] = [int(i) for i in list(instance["pred_classes"].numpy())]

    annotations = list()
    annot_id = 0
    for v in list(
        zip(instance["pred_boxes"], instance["scores"], instance["pred_classes"])
    ):
        annotations.append(
            {
                "id": annot_id,
                "image_id": image_id,
                "bbox": [v[0][0], v[0][1], v[0][2] - v[0][0], v[0][3] - v[0][1]],
                "score": v[1],
                "category_id": v[2],
                "iscrowd": 0,
                "segmentation": [],
            }
        )
        annot_id += 1
    images = list()
    images.append(
        {
            "file_name": file_name,
            "height": d2_instance.image_size[0],
            "width": d2_instance.image_size[1],
            "id": image_id,
        }
    )

    return {"annotations": annotations, "images": images}


def cocoj_get_categories(cocojson):
    """Input is path to a coco (json) file. Return a dictionnary category_id : category_name"""
    with open(cocojson, "r") as f:
        res = {i["id"]: i["name"] for i in json.load(f)["categories"]}
        return res


def crop_annotations(cocodf, input_dir, output_dir):
    """Export each annotation of a coco dataset in separate JPG. cocodf is a dataframe obtained with coco2df"""
    for name in cocodf.name.unique():
        os.makedirs(os.path.join(output_dir, name), exist_ok=True)

    for file in cocodf.file_name.unique():
        im = Image.open(os.path.join(input_dir, file))
        for raw in cocodf[cocodf["file_name"] == file][["box", "id", "name"]].values:
            im.crop(raw[0].bounds).save(
                os.path.join(output_dir, f"{raw[2]}/{raw[1]}.jpg"), "JPEG"
            )
        im.close()


def create_coco_json_for_inference(path_to_inference_folder: str, labels: dir) -> None:
    with open(
        os.path.join(path_to_inference_folder, "inference.json"), "w"
    ) as json_file:
        json_file.write('{\n    "images": [\n')
        valid_input_formats = ("jpg", "jpeg", "png")
        id = 0
        print("-------Inference on these Images-------")
        for file in os.listdir(path_to_inference_folder):
            if file.lower().endswith(valid_input_formats):
                print(f"  {id} {file}")
                im = Image.open(os.path.join(path_to_inference_folder, file))
                width, height = im.size
                if id > 0:
                    json_file.write(",")
                json_file.write("        {\n")
                json_file.write(f'            "file_name": "{file}",\n')
                json_file.write(f'            "height": {height},\n')
                json_file.write(f'            "width": {width},\n')
                json_file.write(f'            "id": {id}\n')
                json_file.write("        }")
                id += 1
        json_file.write("\n    ],\n")
        json_file.write('    "type": "instances",\n')
        json_file.write('    "annotations": [\n')
        json_file.write("    ],\n")
        json_file.write('    "categories": [\n')
        for i, key in enumerate(labels):
            if i > 0:
                json_file.write(",\n")
            json_file.write("        {\n")
            json_file.write('            "supercategory": "none",\n')
            json_file.write(f'            "id": {key},\n')
            json_file.write(f'            "name": "{labels[key]}"\n')
            json_file.write("        }")
        json_file.write("\n    ],\n")
        json_file.write('    "licenses": "",\n')
        json_file.write('    "info": ""\n')
        json_file.write("}")


def sahi_result_to_coco(
    path_to_sahi_json: str, path_to_inference_json: str, path_output_json: str
) -> None:
    with open(path_to_inference_json, "r") as json_file:
        inference = json.load(json_file)
    with open(path_to_sahi_json, "r") as json_file:
        result = json.load(json_file)
    anid = 0
    for d in result:
        d["iscrowd"] = 0
        d["segmentation"] = []
        d["id"] = anid
        anid += 1
    out_json = {}
    out_json["images"] = inference["images"]
    out_json["annotations"] = result
    out_json["categories"] = inference["categories"]
    with open(path_output_json, "w") as json_file:
        json.dump(out_json, json_file, indent=4)
