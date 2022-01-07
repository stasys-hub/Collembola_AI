#!/usr/bin/env python3
"""
Project title:       CollembolAI
Authors:             Stephan Weißbach, Stanislav Sys, Clément Schneider
Original repository: https://github.com/stasys-hub/Collembola_AI.git
Script title:        review_coco_annotations
Script purpose:      Reviewing COCO annotation, please refer to help either calling 
                     "review_coco_annotations.py -h" or directly in the main() function
Usage:               Please refer to help.
Dependencies:        See ReadMe
Last Update:         11.01.2021
"""

# Imports
import argparse
from cocosets_utils import coco2df
import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import PIL
from PIL import Image

PIL.Image.MAX_IMAGE_PIXELS = 500000000

def main():
    
    parser=argparse.ArgumentParser(
        description="""A script that crawl in a COCO annotated dataset, sequentially 
        displaying the content of each bounding box and waiting for the user to input something. The input will be 
        used a new label for the given bounding box. The new annotations are written in a new file, 
        named as the original one but with the additional extension '.reviewed'. The user can interrupt 
        the process using Ctrl + C. It can later be continued at the point of interruption by 
        keeping the original and reviewed files together with their matching names, and calling again on the original file.
        """)

    parser.add_argument('coco_file', type=str, 
            help="Path of the coco file, images must be in the same folder")
    
    args=parser.parse_args()

    
    with open(args.coco_file, 'r') as j:
        r = json.load(j)
        
    df = coco2df(r)

    new_json = args.coco_file + '.reviewed'
    try:
        with open(new_json, 'r') as j:
            nr = json.load(j)
            n_df = coco2df(nr)
    except:
        nr = r
        nr['annotations'] = []
        n_df = pd.DataFrame()
        
    try:
        done_ids = n_df['id'].values()
    except:
        done_ids = []
    
    df = df[~df['id'].isin(done_ids)]

    for file in df.file_name.unique():
        im = Image.open(os.path.dirname(args.coco_file) + '/' + file)
        for raw in df[df['file_name'] == file][['box', 'id', 'area', 'bbox', 'image_id']].values:
            plt.imshow(im.crop(raw[0].bounds))
            plt.show(block=False)
            inp = input("annotation")
            nr['annotations'].append(            
                        {'area': raw[2],
                         'iscrowd': 0,
                         'bbox': raw[3],
                         'category_id': inp,
                         'ignore': 0,
                         'segmentation': [],
                         'image_id': raw[4],
                         'id': raw[1]})
            plt.close() # will make the plot window empty
            with open(new_json, 'w') as j:
                json.dump(nr, j)
        im.close()


if __name__ == "__main__":
    main()
