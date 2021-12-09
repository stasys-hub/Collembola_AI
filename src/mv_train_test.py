#!/usr/bin/env python3
"""
Project title:       CollembolAI
Authors:             Stephan Weißbach, Stanislav Sys, Clément Schneider
Original repository: https://github.com/stasys-hub/Collembola_AI.git
Module title:        mv_train_test
Purpose:             Coco converted from PascalVOC annotations generated with labelimg, using voc2coco.py
                     are missing some non-vital fields, have non integer image id (some downstream scripts expect int), and include the full images file path
                     that is annoying when moving the folders to work places. This script fix those issues IN PLACE.
Dependencies:        See ReadMe
Last Update:         
Licence:             
"""

import argparse
import json
import os
import shutil
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()

    parser.add_argument('coco_json', type=str,
            help='''Path of the coco file''')

    args=parser.parse_args()

    with open(args.coco_json, 'r') as j:
        coco = json.load(j)

    destination = Path(args.coco_json).with_suffix('')
    destination.mkdir(parents=True, exist_ok=True)

    for i in coco['images']:
        shutil.move(i['file_name'], destination)

    shutil.move(args.coco_json, destination)

if __name__ == "__main__":
    main()

