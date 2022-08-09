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

import json
import os
import pandas as pd
from PIL import Image
from shapely.geometry import box


def testresults2coco(test_dir, inference_dir, write=False):
    '''
    Returns a full Coco instance from the test results annotations file (json outputed by Detectron 2)
    It will use the test COCO annotation file to fetch missing informations. If write parameter is set to True, 
    then write the COCO to json in the test results directory
    '''
    
    # Open ground truth test annotations (COCO)
    with open(f'{test_dir}/test.json', 'r') as j:
        ttruth = json.load(j)

    # Open inference results and make it a complete COCO format by replacing the ttruth COCO annotations
    # and creating other standard fields (not all useful to us, but could allow to process it with further "COCO tools")
    tinference = ttruth.copy()
    with open(f'{inference_dir}/result.json', 'r') as j:
        tinference['annotations'] = json.load(j)
        
    anid = 0
    for d in tinference['annotations']:
        d['iscrowd'] = 0
        d['segmentation'] = []
        d['id'] = anid
        anid += 1
    
    # Write inference results in COCO format json
    if write:
        with open(f'{inference_dir}/result_output.json', 'w') as j:
            json.dump(tinference, j)
    
    return tinference

def COCObox_2_shapely(x):  
    '''convert COCO bbox coordinates style to shapely box coordinates style
    x: COCO bbox coordinates (list of 4 elements)'''
    return box(x[0],x[1],x[0]+x[2],x[1]+x[3])

def coco2df(coco):
    '''
    Fit a coco instance into a flat pandas DataFrame. Likely overkill to bring in pandas for this,
    but considerably simply operations from my own perspective =).
    '''
    classes_df = pd.DataFrame(coco['categories'])
    classes_df.name = classes_df.name.str.strip()
    classes_df = classes_df.rename(columns={"id": "category_id"})
    # classes_df.rename(columns={"id": "category_id"}, inplace=True)
    images_df = pd.DataFrame(coco['images'])
    images_df.rename(columns={"id": "image_id"}, inplace=True)
    coco_df = pd.DataFrame(coco['annotations'])\
                    .merge(classes_df, on="category_id", how='left')\
                    .merge(images_df, on="image_id", how='left')
    
    coco_df['box'] = coco_df['bbox'].apply(COCObox_2_shapely)
    coco_df['area'] = coco_df['box'].apply(lambda x: x.area)

    return coco_df

def df2coco(df):
    '''
    Take a dataframe and make it into a coco instance, provided the correct fields are present
    '''
    coco = dict()
    coco['images'] = df[['file_name', 'height', 'width', 'image_id']].rename(columns={'image_id':'id'}).drop_duplicates(subset=['file_name']).to_dict(orient="record")
    coco['annotations'] = df[['area', 'iscrowd', 'bbox', 'category_id', 'segmentation', 'image_id', 'id']].to_dict(orient="record")
    coco['categories'] = df[['supercategory', 'category_id', 'name']].rename(columns={'category_id':'id'}).drop_duplicates(subset=['name']).to_dict(orient="record")
    coco['type'] = 'instances'
    coco['licenses'] = ''
    coco['info'] = ''
    return coco

def extract_classes_from_coco_json(path_to_coco_json: str, output_path: str) -> None:
    '''
    Creates a JSON file with classes and IDs from a valid coco json.
    '''
    with open(path_to_coco_json,"r") as input_coco_json:
        coco_json = json.load(input_coco_json)
    classes_dict = {}
    for _, row in enumerate(coco_json["categories"]):
        classes_dict[str(row['id'])] = row['name']
    with open(output_path, "w") as out_classes:
        json.dump(classes_dict,out_classes)
    

def sahi_result_to_coco(path_to_sahi_json: str, path_to_inference_json: str, path_output_json: str) -> None:
    '''
    Reformat the SAHI output to a full COCO Json.
    '''
    with open(path_to_inference_json, "r") as json_file:
        inference = json.load(json_file)     
    with open(path_to_sahi_json,"r") as json_file:
        result = json.load(json_file)
    anid = 0
    for d in result:
        d['iscrowd'] = 0
        d['segmentation'] = []
        d['id'] = anid
        anid += 1
    out_json = {}
    out_json["images"] = inference["images"]
    out_json["annotations"] = result
    out_json["categories"] = inference["categories"]
    with open(path_output_json,"w") as json_file:
        json.dump(out_json, json_file, indent=4)


def create_coco_json_for_inference(path_to_inference_folder: str, labels:dict) -> None:
    '''
    SAHI operates only when a reference JSON for the inference folder is given. This
    function creates such an JSON, listing the images and the respective width, height.
    '''
    Image.MAX_IMAGE_PIXELS = None
    with open(os.path.join(path_to_inference_folder,"inference.json"),"w") as json_file:
        json_file.write('{\n    "images": [\n')
        valid_input_formats = ("jpg","jpeg","png")
        id = 0
        print("-------Inference on these Images-------")
        for file in os.listdir(path_to_inference_folder):
            if file.lower().endswith(valid_input_formats):
                print(f"  {id} {file}")
                im = Image.open(os.path.join(path_to_inference_folder,file))
                width, height = im.size
                if id > 0:
                    json_file.write(',')
                json_file.write('        {\n')
                json_file.write(f'            "file_name": "{file}",\n')
                json_file.write(f'            "height": {height},\n')
                json_file.write(f'            "width": {width},\n')
                json_file.write(f'            "id": {id}\n')
                json_file.write('        }')
                id += 1
        json_file.write('\n    ],\n')
        json_file.write('    "type": "instances",\n')
        json_file.write('    "annotations": [\n')
        json_file.write('    ],\n')
        json_file.write('    "categories": [\n')
        for i,key in enumerate(labels):
            if i>0:
                json_file.write(',\n')
            json_file.write('        {\n')
            json_file.write('            "supercategory": "none",\n')
            json_file.write(f'            "id": {key},\n')
            json_file.write(f'            "name": "{labels[key]}"\n')
            json_file.write('        }')
        json_file.write('\n    ],\n')
        json_file.write('    "licenses": "",\n')
        json_file.write('    "info": ""\n')
        json_file.write('}')
