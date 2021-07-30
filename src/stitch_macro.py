#!/usr/bin/env python3

"""
Project title:       CollembolAI
Authors:             Stephan Weißbach, Stanislav Sys, Clément Schneider
Original repository: https://github.com/stasys-hub/Collembola_AI.git
Script title:        stitch_macro
Script purpose:      Stitching sets of overlapping pictures (in our case, taken with our
                     hoverMacroCam system).
Usage:               Please refer to help.
Dependencies:        See ReadMe
Last Update:         11.01.2021
"""

import argparse
import cv2
import os
import re

def loop_stitch(in_folder, out_folder, out_jpeg_quality=92):    
    imset_folders = next(os.walk(f'{in_folder}'))[1]
    print(f'Found {len(imset_folders)} subfolders in the input folder. Will try to find and stitch pictures' +
         ' in each of those.')
    for imset in imset_folders:
        stitch(in_folder.rstrip('/')+'/'+imset, out_folder, out_jpeg_quality)
       

def stitch(imset, out_folder, out_jpeg_quality=92):
    """Stitching the pictures
    in_folder: folder that contains the pictures to be stiched. All pictures found in 
    the folder are going to be loaded. The script will select pictures based on presence of common pictures extension 
    in the file names (jpg, png and tiff).
    outfile: filename for the output stitched picture."""
    
    set_name = os.path.basename(imset)
    images_extension = "(jpg|JPEG|tif|tiff|TIF|TIFF|png|PNG)"
    
    print(f'{set_name}: loading images')
    images = [cv2.imread(imset +'/'+image) for image in os.listdir(imset) if re.match(f'.*\.{images_extension}', image)]
    if len(images) > 0:
        print(f'{set_name}: {len(images)} images found, stitching now')
        stitcher = cv2.createStitcher() if cv2.__version__.startswith('3') else cv2.Stitcher_create(mode=1)
        (status, newseed) = stitcher.stitch(images)
        if int(status) == 0:
            out_file = f'{out_folder.rstrip("/")}/{set_name}.jpg'
            print(f'{set_name}: stitcher report SUCCESS (= at least 2 pictures could be stitched)')
            print(f'{set_name}: writing result to {out_file}')
            cv2.imwrite(out_file, newseed, [int(cv2.IMWRITE_JPEG_QUALITY), out_jpeg_quality])
            
        else:
            print(f'{set_name}: stitcher failed for some reason, with error code {status}. Please refer to opencv2 createStitcher documentation '+
                 'for more details.')
            
def main():
    
    parser=argparse.ArgumentParser()

    parser.add_argument('in_folder', type=str, 
            help='''Input folder. Must contains sets of pictures to stitch organized as direct subfoler, or being itself a pictures set (for single set mode''')

    parser.add_argument('-o', type=str, default='./',
            help='''Output directory. Will use the working directory by default''')    

    parser.add_argument('-q', type=int, default=92,
            help='''Quality of the jpg output. Default = 92''')

    parser.add_argument('-s', action='store_true',
            help='''If provided, will treat the input folder as a single pictures set.''')


    args=parser.parse_args()
    if args.s:
        stitch(args.in_folder.rstrip('/'), args.o, args.q)
    else:
        loop_stitch(args.in_folder, args.o, args.q)


if __name__ == "__main__":
    main()

