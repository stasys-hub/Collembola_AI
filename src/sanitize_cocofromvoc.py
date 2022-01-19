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
import argparse
from cocosets_utils import numerize_img_id, numerize_annot_id, numerize_cat_id,\
                           trim_path_from_file_name, add_standard_field, drop_category
        
def main():
    
    parser=argparse.ArgumentParser()

    parser.add_argument('coco_json', type=str, 
            help='''Path of the coco file to sanitize''')
    
    parser.add_argument('--drop_cat', type=int, default=-1,
            help='''Drop categorie and related annotations based on provided categorie id. 
                    Default do not drop anything''')
    
    parser.add_argument('-i', '--inplace',action='store_true',
            help='''Works in place, no backup. Default: a new file is created in the same directory ''')
    
    args=parser.parse_args()
  
    if args.inplace:
        outputf = args.coco_json
    else:
        outputf = args.coco_json + '.sanitized'
        
    with open(args.coco_json, 'r') as j:
        coco = json.load(j)
     
    numerize_img_id(coco)
    numerize_cat_id(coco)
    
    if int(args.drop_cat) > -1:
        drop_category(coco, int(args.drop_cat))
    
    numerize_annot_id(coco)
    trim_path_from_file_name(coco)
    add_standard_field(coco)
    
    with open(outputf, 'w', encoding='utf-8') as j:
        json.dump(coco, j, ensure_ascii=False, indent=4)
    

if __name__ == "__main__":
    main()
