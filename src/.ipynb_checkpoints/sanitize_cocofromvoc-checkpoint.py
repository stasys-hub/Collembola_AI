#!/usr/bin/env python3
"""
Project title:       CollembolAI
Authors:             Stephan Weißbach, Stanislav Sys, Clément Schneider
Original repository: https://github.com/stasys-hub/Collembola_AI.git
Module title:        sanitize_cocofromvoc.py
Purpose:             Coco converted from PascalVOC annotations generated with labelimg, using voc2coco.py
                     are missing some non-vital fields, have non integer image id (some downstream scripts expect int), and include the full images file path
                     that is annoying when moving the folders to work places. This script fix those issues IN PLACE.
Dependencies:        See ReadMe
Last Update:         
Licence:             
"""

import json
import ntpath
import argparse

def numerize_img_id(coco):
    # replace string image ids by integer ids
    new_id = 0
    id_mapper = dict()
    for i in coco['images']:
        id_mapper[i['id']] = new_id
        i['id'] = new_id
        new_id += 1
        
    # remap annotation
    for a in coco['annotations']:
        a['image_id'] = id_mapper[a['image_id']]

def numerize_annot_id(coco):
    # reset annotation ids by integer ids and enforce numerical value on categories id, but only if numerical character
    new_id = 1
    for i in coco['annotations']:
        i['id'] = new_id
        try:
            i['category_id'] = int(i['category_id'])
        except:
            pass
        new_id += 1

def numerize_cat_id(coco):
    # enforce numerical value on categories id, but only if numerical character
    for i in coco['annotations']:
        try:
            i['id'] = int(i['id'])
        except:
            pass

def trim_path_from_file_name(coco):
    for i in coco['images']:
        i['file_name'] = ntpath.basename(i['file_name'])
        
def add_standard_field(coco):
    try: 
        coco['licenses']
    except:
        coco['licenses'] = ''
    try: 
        coco['info']
    except:
        coco['info'] = ''
        
def main():
    
    parser=argparse.ArgumentParser()

    parser.add_argument('coco_json', type=str, 
            help='''Path of the coco file to sanitize''')
    
    args=parser.parse_args()
        
    with open(args.coco_json, 'r') as j:
        coco = json.load(j)
   
    numerize_img_id(coco)
    numerize_annot_id(coco)
    numerize_cat_id(coco)
    trim_path_from_file_name(coco)
    add_standard_field(coco)
    
    with open(args.coco_json, 'w', encoding='utf-8') as j:
        json.dump(coco, j, ensure_ascii=False, indent=4)
    

if __name__ == "__main__":
    main()
