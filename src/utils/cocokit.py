#!/usr/bin/env python3
"""
Project title:       CollembolAI
Authors:             Stephan Weißbach, Stanislav Sys, Clément Schneider
Original repository: https://github.com/stasys-hub/Collembola_AI.git
Script title:        cocokit
Script purpose:      Some tools for handling json coco files, please refer to help either calling 
                     "cocokit.py -h" or directly in the main() function
Usage:               Please refer to help.
Dependencies:        See ReadMe
Last Update:         11.01.2021
"""

# Standard
import argparse
import json
import os
from pathlib import Path
import random
import sys
import rglob

# Third party
import matplotlib.pyplot as plt
import pandas as pd
import PIL
from PIL import Image

# CollembolAI
from cocosets_utils import coco2df, numerize_img_id, numerize_annot_id, numerize_cat_id,\
                           trim_path_from_file_name, add_standard_field, drop_category,\
                           mergecocos

# Allows loading large images
PIL.Image.MAX_IMAGE_PIXELS = 500000000

def main():
    
    
    parser=argparse.ArgumentParser(
        description="""
        Some tools for handling json coco files, used when building the dataset.
        """, 
        epilog="""
                Example of use:\n
                Merging two coco datasets: cocokit.py -m cocofile1.json cocofile2.json\n
                Drop categorie (id=0) with its annotations: cocokit.py --drop_cat 0 cocofile.json\n
                Improve/fix coco formating: cocokit.py --f cocofile.json\n
                Review annotations: cocokit.py -r cocofile.json\n
                """)
    
    
    parser.add_argument('coco_json', type=str, 
            help='''Path of the coco file (json) in input, the related images are expected to be in the same folder''')
    
    parser.add_argument('-m', '--merge', type=str, default=None,
            help='''Merge two coco datasets, the output merged file is created in the execution directory''')
       
    parser.add_argument('--drop_cat', type=int, default=-1,
            help='''Drop categorie and related annotations based on provided categorie id. 
                    Default do not drop anything''')
    
    parser.add_argument('-f', '--format', action='store_true',
            help='''Fix some inconsistencies in the coco file, that may results from manual operation or 
            use of other tools. Also indent the json to facilitate reading.''')
    
    parser.add_argument('-i', '--inplace', action='store_true',
            help='''Works in place, no backup. Default: a new file is created in the same directory. Ignored by --merge and --review that will always create a new file''')
    
    
    parser.add_argument('-s', '--split', action='store_true',
            help='''Splitting the COCO JSON file given in input. The script will look for pictures in pre-existing test and train folders in the same directory as the input, then split the COCO JSON accordingly''') 

    parser.add_argument('--splitratio', type=int, default=0,
                        help='''Splitting the COCO JSON file given in input according to a percentage ratio. Take an integer value that is the percentage of the total dataset to go in the test set when using the splitting function. Remaining goes in the train set.''')
    
    parser.add_argument('-r', '--review', action='store_true',
            help='''Sequentially displays the content of each annotations and waits for user's input. 
            The input serve as new label ID for the given bounding box. The modified annotations are written in a new coco json file, named after the original one with the additional extension '.reviewed'. The process can be interrupted at any time, then resumed at the point of interruption using the same command, as long as the original and reviewed files are kept together''')
    
    args=parser.parse_args()
    
    with open(args.coco_json, 'r') as j:
        coco = json.load(j)
        
    worksinplace = args.inplace
    output_file = args.coco_json
    
    # In case of merging
    if args.merge:
        with open(args.merge, 'r') as j:
            coco2 = json.load(j)
        
        coco = mergecocos(coco, coco2)
        worksinplace = False
        output_file = output_file + '.merged'
        
    if args.format:
        
        if worksinplace:
            output_file = args.coco_json
        else:
            output_file = output_file + '.formated'
        
        numerize_img_id(coco)
        numerize_cat_id(coco)
    
        if int(args.drop_cat) > -1:
            drop_category(coco, int(args.drop_cat))
    
        numerize_annot_id(coco)
        trim_path_from_file_name(coco)
        add_standard_field(coco)
    
    with open(output_file, 'w', encoding='utf-8') as j:
        json.dump(coco, j, ensure_ascii=False, indent=4)
        
    sys.exit()
    
    
    if args.review:
        print("cocokit executed in reviewing mode")
        
        df = coco2df(coco)

        output_file = args.coco_json + '.reviewed'
        print('Trying to open a previously saved file:  {}'.format(output_file))
        try:
            with open(output_file, 'r') as j:
                nr = json.load(j)
                n_df = coco2df(nr)
        except Exception as e:
            print('No valid saved file, a new file {} will be created'.format(output_file))
            nr = r
            nr['annotations'] = []
            n_df = pd.DataFrame()
        
        try:
            done_ids = n_df['id'].values
        except:
            done_ids = []

        df = df[~df['id'].isin(done_ids)]
    
        remaining_num = df.shape[0]
        print ('Still {} annotations to check :)'.format(remaining_num))

        for file in df.file_name.unique():
            im = Image.open(os.path.join(os.path.dirname(args.coco_json), file))
            for raw in df[df['file_name'] == file][['box', 'id', 'area', 'bbox', 'image_id']].values:
                remaining_num = remaining_num - 1
                plt.imshow(im.crop(raw[0].bounds))
                plt.show(block=False)
                inp = input("annotation ID (should be integer, {} remains): ".format(remaining_num))
                try:
                    inp = int(inp)
                except:
                    pass
                nr['annotations'].append(            
                        {'area': raw[2],
                         'iscrowd': 0,
                         'bbox': raw[3],
                         'category_id': inp,
                         'ignore': 0,
                         'segmentation': [],
                         'image_id': raw[4],
                         'id': raw[1]})
                plt.close()
                with open(output_file, 'w') as j:
                    json.dump(nr, j, indent=4)
            im.close()
        
        print('All done :)')
        sys.exit()
        

    
    if args.splitratio > 0 and args.splitratio < 100: 
        
        dirname = Path(args.coco_json).parents[0]
             
        print(f'Splitting: {args.splitratio} % of the pictures are going to the test split')          
        img_list = [i['file_name'] for i in coco['images']]
        cut = int(len(img_list) * args.ratio / 100)
        random.shuffle(img_list)
        test_coco = reduce_coco(coco, img_list[:cut])
        train_coco = reduce_coco(coco, img_list[cut:])
        
        print('Moving files to test dir')
        
        os.makedirs(f'{dirname}/test', exist_ok=True)
        with open(f'{dirname}/test/test.json', 'w') as j:
            json.dump(test_coco, j, indent=4)
        for img in img_list[:cut]:
            shutil.move(f'{dirname}/{img}', f'{dirname}/test/{img}')
            
        print('Moving files to train dir')
        os.makedirs(f'{dirname}/train', exist_ok=True)
        with open(f'{dirname}/train/train.json', 'w') as j:
            json.dump(train_coco, j, indent=4)     
        for img in img_list[cut:]:
            shutil.move(f'{dirname}/{img}', f'{dirname}/train/{img}')
            
        print('Done')
        sys.exit()
        
    if args.split:
    
        dirname = Path(args.coco_json).parents[0]
        
        print('Will look for pre-existing train and test folders and split the COCO JSON based on the folers content')
        
        try:
            test_img_list = [i.name for i in Path(dirname + '/test.').rglob('*.jpg')]
            if len(test_img_list) <= 0:
                print('Nothing in the test folder')
            else:
                test_coco = reduce_coco(coco, test_img_list)
                print('Writing test.json in test dir')
                with open(f'{dirname}/test/test.json', 'w') as j:
                    json.dump(test_coco, j, indent=4)   
        except:
            print('No valid test folder')
            
            
        try:
            train_img_list = [i.name for i in Path(dirname + '/train.').rglob('*.jpg')]
            if len(train_img_list) <= 0:
                print('Nothing in the test folder')
            else:
                test_coco = reduce_coco(coco, train_img_list)
                print('Writing train.json in test dir')
                with open(f'{dirname}/train/train.json', 'w') as j:
                    json.dump(train_coco, j, indent=4)   
        except:
            print('No valid test folder')
            
        print('Done')
        sys.exit()


if __name__ == "__main__":
    main()

