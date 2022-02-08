"""
Project title:       CollembolAI
Authors:             Stephan Weißbach, Stanislav Sys, Clément Schneider
Original repository: https://github.com/stasys-hub/Collembola_AI.git
Module title:        cocosets_utils
.py
Purpose:             a set of inhouse utilitary functions, related to operations on handling the dataset
                     of COCO formatted annotations.
Dependencies:        See ReadMe
Last Update:         11.01.2021
"""

from itertools import combinations, product
import json
import numpy as np
import ntpath
import os
import pandas as pd
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from shapely.geometry import box
import random

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
    with open(f'{inference_dir}/test_results/coco_instances_results.json', 'r') as j:
        tinference['annotations'] = json.load(j)
        
    anid = 0
    for d in tinference['annotations']:
        d['iscrowd'] = 0
        d['segmentation'] = []
        d['id'] = anid
        anid += 1
    
    # Write inference results in COCO format json
    if write:
        with open(f'{inference_dir}/test_results/results.json', 'w') as j:
            json.dump(tinference, j)
    
    return tinference

def getbox(x):  
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
    classes_df.rename(columns={"id": "category_id"}, inplace=True)
    images_df = pd.DataFrame(coco['images'])
    images_df.rename(columns={"id": "image_id"}, inplace=True)
    coco_df = pd.DataFrame(coco['annotations'])\
                    .merge(classes_df, on="category_id", how='left')\
                    .merge(images_df, on="image_id", how='left')
    
    coco_df['box'] = coco_df['bbox'].apply(getbox)
    coco_df['area'] = coco_df['box'].apply(lambda x: x.area)

    return coco_df

def draw_coco_bbox(coco, out_dir, coco_dir, prefix='annotated', line_width=10, fontsize = 80, fontYshift = -50):
    '''
    Detectron2 module for writing annotated pictures was not so explicit to me, and default output not so pretty.
    This function will draw the annotation on the pictures of a coco dataset. The dataset can be provided as a coco instance,
    or as a dataframe resulting from coco2df. Modified pictures are written to the out_dir, with a name prefix.
    To adjust display, simply change line_width (= box line), font_size (= label font). Labels text can be shifted vertically
    with fontYshift.
    '''
    
    colors = ['aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 
              'beige', 'bisque', 'blanchedalmond', 'blue', 'blueviolet', 
              'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 
              'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'deeppink', 'deepskyblue', 
              'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 
              'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'gray', 'grey', 'green', 
              'greenyellow', 'honeydew', 'hotpink', 'indianred', 'ivory', 'khaki', 'lavender', 
              'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 
              'lightgoldenrodyellow', 'lightgreen', 'lightgray', 'lightgrey', 'lightpink', 'lightsalmon', 
              'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 
              'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine', 'mediumblue', 
              'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 
              'mediumvioletred', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'oldlace', 'olive', 'olivedrab', 'orange', 
              'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 
              'peru', 'pink', 'plum', 'powderblue', 'purple', 'rebeccapurple', 'red', 'rosybrown', 'royalblue', 'saddlebrown', 
              'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'slategrey', 
              'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white', 
              'whitesmoke', 'yellow', 'yellowgreen']
    scolors = random.sample(colors, len(colors))
    
    fnt = ImageFont.truetype(os.path.dirname(os.path.realpath(__file__)) + "/FreeMono.ttf", fontsize)
    
    try:
        coco_df = coco2df(coco)
    except:
        coco_df = coco
    
    if 'score' in coco_df.columns:
        coco_df['name'] = coco_df['name'] + ' ' + round(coco_df['score'], 2).astype(str)
        
    if 'is_false_positive' in coco_df.columns:
        coco_df['name'] = coco_df['name'] + ' ' + (~coco_df['is_false_positive']).astype(str) + ' detection'
    
    resh = lambda x : ((x[0],x[1]), (x[0]+x[2],x[1]+x[3]))
    coco_df['coordinates'] = coco_df['bbox'].apply(resh)
    
    coco_df['color'] = ""
    for c in coco_df.name.unique():
        if len(scolors) == 0:
            scolors = random.sample(colors, len(colors))
        color = scolors.pop()
        coco_df['color'] = coco_df['color'].where(~(coco_df['name'] == c), color)
        
    for img_name in coco_df.file_name.unique():

        if len(scolors) == 0:
            scolors = random.sample(colors, len(colors))
        outline = scolors.pop()
        source_img = Image.open(f'{coco_dir}/{img_name}')
        draw = ImageDraw.Draw(source_img)
        for row in coco_df[coco_df['file_name'] == img_name][['name','coordinates', 'color']].values:
            draw.rectangle(row[1], outline=row[2], width=line_width)
            draw.text((row[1][0][0], row[1][0][1]+fontYshift), row[0], font=fnt, fill=row[2])
        
        print(f'Writing {out_dir}/{prefix}_{img_name}')
        source_img.save(f'{out_dir}/{prefix}_{img_name}', "JPEG")

           
def deduplicate_overlapping_preds(df_pred, IoU_threshold=0.7, area=1000000):
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

def match_true_n_pred_box(df_ttruth, df_pred, IoU_threshold=0.4):
    matched = pd.DataFrame()
    df_pred['id_pred'] = df_pred['id']
    df_pred['pred_box'] = df_pred['box']
    df_ttruth['id_true'] = df_ttruth['id']
    df_ttruth['true_box'] = df_ttruth['box']
    df_ttruth['true_area'] = df_ttruth['area']
    
    
    for image_id in df_pred.image_id.unique():
        sdf_pred = df_pred[df_pred['image_id'] == image_id]
        sdf_ttruth = df_ttruth[df_ttruth['image_id'] == image_id]
        sdf_ttruth = df_ttruth[df_ttruth['image_id'] == image_id]

        df = pd.DataFrame(product(sdf_ttruth['id'], sdf_pred['id']), columns=['id_true', 'id_pred'])
        df = df.merge(sdf_ttruth[['id_true', 'true_box', 'true_area']], how='left', on='id_true')\
            .merge(sdf_pred[['id_pred', 'pred_box', 'score']], how='left', on='id_pred')

        df['intersection'] = df[['true_box', 'pred_box']].apply(lambda x: x[0].intersection(x[1]).area, axis=1)
        df['union'] = df[['true_box', 'pred_box']].apply(lambda x: x[0].union(x[1]).area, axis=1)
        df['IoU'] = df['intersection'] / df['union']
        matched = pd.concat([matched, df], axis=0)
        
    df2 = matched[matched['IoU'] > IoU_threshold].sort_values(by='score', ascending = False)
    df2 = df2.drop_duplicates(subset=['id_pred'], keep='first')
    df2 = df2.drop_duplicates(subset=['id_true'], keep='first')

    
    pairs = df_ttruth[['id_true', 'name']].merge(df2[['id_true', 'id_pred']], how='left', on='id_true')\
        .merge(df2[['id_pred', 'score']], how='outer', on='id_pred')\
        .rename(columns={"name": "name_true"})

    pairs = pairs.merge(df_pred[['id_pred', 'name']], how='outer', on='id_pred')\
                 .rename(columns={"name": "name_pred"})

    pairs['is_correct'] = (pairs['name_true'] == pairs['name_pred'])
    pairs['is_correct_class'] = (pairs['name_true'] == pairs['name_pred']).where(pairs.id_pred.notnull(), np.nan)
        
    return pairs

def numerize_img_id(coco):
    '''
    Input: coco instance (dict)
    coco instance is modified in-place
    '''
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
    '''
    Input: coco instance (dict)
    coco instance is modified in-place
    '''
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
    '''
    Input: coco instance (dict)
    coco instance is modified in-place
    '''
    # enforce numerical value on categories id, but only if numerical character
    for i in coco['annotations']:
        try:
            i['id'] = int(i['id'])
        except:
            pass

def trim_path_from_file_name(coco):
    '''
    Input: coco instance (dict)
    coco instance is modified in-place
    '''
    for i in coco['images']:
        i['file_name'] = ntpath.basename(i['file_name'])
        
def add_standard_field(coco):
    '''
    Input: coco instance (dict)
    coco instance is modified in-place
    '''
    try: 
        coco['licenses']
    except:
        coco['licenses'] = ''
    try: 
        coco['info']
    except:
        coco['info'] = ''
        
        
def drop_category(coco, cat_id):
    '''
    Input: coco instance (dict) and categorie_id (integer)
    coco instance is modified in-place
    ''' 
    coco['annotations'] = [i for i in coco['annotations'] if i['category_id'] != cat_id]
    coco['categories'] = [i for i in coco['categories'] if i['id'] != cat_id]

    
def mergecocos(coco1_i, coco2_i):
    '''
    Input: two coco instances (dict). Return merged coco instance.
    ''' 
    coco1 = coco1_i.copy()
    coco2 = coco2_i.copy()
    #make image_id unique
    for i in coco1['images']:
        i['id'] = 'coco1_'+str(i['id'])
    for i in coco1['annotations']:
        i['image_id'] = 'coco1_'+str(i['image_id'])
        
    for i in coco2['images']:
        i['id'] = 'coco2_'+str(i['id'])
    for i in coco2['annotations']:
        i['image_id'] = 'coco2_'+str(i['image_id'])

    cocomerged = coco1.copy()
    cocomerged['images'] = coco1['images'] + coco2['images']
    cocomerged['annotations'] = coco1['annotations'] + coco2['annotations']
    cocomerged['categories'] = cocomerged['categories']

    a_id = 1
    for i in cocomerged['annotations']:
        i['id'] = a_id
        a_id += 1

    numerize_img_id(cocomerged)
    
    return cocomerged


def make_a_random_box(max_ratio, min_ratio, max_length, min_length, img_width, img_height):
    height = int(random.uniform(min_length, max_length))
    width = int(height * random.uniform(0.5, 1.5))
    x= int(random.uniform(0, img_width-width))
    y= int(random.uniform(0, img_height-height))
    rbox = geo.Polygon([[x,y],[x+width,y],[x+width, y+height], [x, y+height]])
    return rbox


def extract_random_background_subpictures(coco_df, pictures_dir, output_dir, num_subpict_per_pict=200):
    
    # Get min ratio and max ratio of annotation in the train set
    max_ratio = coco_df.bbox.apply(lambda x:x[3]/x[2]).max()
    min_ratio = coco_df.bbox.apply(lambda x:x[3]/x[2]).min()
    max_length = int(coco_df.bbox.apply(lambda x:x[2]).max() / 2) #division by 2 because large background boxes are not really a thing.
    min_length = coco_df.bbox.apply(lambda x:x[2]).min()

    bnum = 0

    for file in train.file_name.unique():
    
        num_pict = 0
        subcoco_df = coco_df[coco_df['file_name']==file]
    
        img_width = subcoco_df.width.values[0]
        img_height = subcoco_df.height.values[0]
        
        im = Image.open(pictures_dir + '/' + file)
        
        while num_pict < num_subpict_per_pict:
            rbox = make_a_random_box(max_ratio, min_ratio, max_length, min_length, img_width, img_height)
                #if not overlapping with a specimen
            if not subcoco_df['box'].apply(lambda x:x.intersects(rbox)).any():
                im.crop(rbox.bounds).save(f'{output_dir}/background_{bnum}.jpg','JPEG')
                bnum += 1
                num_pict += 1
        print(f'{num_pict} written for {file}')

def d2_instance2dict(d2_instance, image_id, file_name):
    '''
    Convert a detectron2 instance into a JSON serializable dict (COCO annotations format).
    '''
    instance = dict(d2_instance.get_fields())
    instance['image_size'] = d2_instance.image_size
    
    pb = list()
    for i in instance['pred_boxes'].tensor:
        box = [int(j) for j in np.round(i.numpy()).astype('int')]
        pb.append(box)

    instance['pred_boxes'] = pb
    instance['scores'] = [round(float(i), 4) for i in instance['scores'].numpy()]
    instance['pred_classes'] = [int(i) for i in list(instance['pred_classes'].numpy())]

    annotations = list()
    annot_id = 0
    for v in list(zip(instance['pred_boxes'], instance['scores'], instance['pred_classes'])):
        annotations.append(
                {'id': annot_id,
                 'image_id': image_id, 
                 'bbox': [v[0][0], v[0][1], v[0][2]-v[0][0], v[0][3]-v[0][1]], 
                 'score': v[1], 
                 'category_id': v[2],
                 'iscrowd':0,
                 'segmentation': []}
                )
        annot_id += 1
    images = list()
    images.append(
            {'file_name': file_name,
             'height': d2_instance.image_size[0],
             'width': d2_instance.image_size[1],
             'id': image_id}
            )

    return {'annotations': annotations, 'images':images}

def cocoj_get_categories(cocojson):
    """ Input is path to a coco (json) file. Return a dictionnary category_id : category_name """
    with open(cocojson, 'r') as f:
        res = {i['id']: i['name'] for i in json.load(f)['categories']}
        return res

    
def crop_annotations(cocodf, input_dir, output_dir):
    for name in cocodf.name.unique():     
        os.makedirs(os.path.join(output_dir, name), exist_ok=True)   
        
    for file in cocodf.file_name.unique():
        im = Image.open(os.path.join(input_dir, file))
        for raw in cocodf[cocodf['file_name']==file][['box', 'id', 'name']].values:
            im.crop(raw[0].bounds).save(os.path.join(output_dir, f'{raw[2]}/{raw[1]}.jpg'),'JPEG')
        im.close()

