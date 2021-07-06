from itertools import combinations, product
import json
import numpy as np
import os
import pandas as pd
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from shapely.geometry import box
import random


def testresults2coco(test_dir, inference_dir, write=False):
    '''
    Return a full Coco instance from the test results annotations
    If write parameter is set to True, then write the COCO to json in the test results directory
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


def coco2df(coco):
    '''
    fit a coco instance into a flat pandas DataFrame. Solely to make it more comfortable for me to handle.
    '''
    classes_df = pd.DataFrame(coco['categories'])
    classes_df.name = classes_df.name.str.strip()
    classes_df.columns = ['supercategory','category_id','name']
    images_df = pd.DataFrame(coco['images'])
    images_df.columns = ['file_name','height','width','image_id']
    coco_df = pd.DataFrame(coco['annotations'])\
                    .merge(classes_df, on="category_id", how='left')\
                    .merge(images_df, on="image_id", how='left')
    
    getbox = lambda x : box(x[0],x[1],x[0]+x[2],x[1]+x[3])
    coco_df['box'] = coco_df['bbox'].apply(getbox)
    coco_df['area'] = coco_df['box'].apply(lambda x: x.area)

    return coco_df

def draw_coco_bbox(coco, out_dir, test_dir, prefix='annotated', line_width=10, fontsize = 80, fontYshift = -50):
    '''
    
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
        coco_df['name'] = coco_df['name'] + ' ' + coco_df['is_false_positive'].astype(str)
    
    resh = lambda x : ((x[0],x[1]), (x[0]+x[2],x[1]+x[3]))
    coco_df['coordinates'] = coco_df['bbox'].apply(resh)
    
    coco_df['color'] = ""
    for c in coco_df.name.unique():
        if len(scolors) == 0:
            scolors = random.sample(colors, len(colors))
        color = scolors.pop()
        coco_df['color'] = coco_df['color'].where(~(coco_df['name'] == c), color)

    for img_name in coco_df.file_name.unique():
        print(f'Drawing on {img_name}')
        if len(scolors) == 0:
            scolors = random.sample(colors, len(colors))
        outline = scolors.pop()
        source_img = Image.open(f'{test_dir}/{img_name}')
        draw = ImageDraw.Draw(source_img)
        for row in coco_df[coco_df['file_name'] == img_name][['name','coordinates', 'color']].values:
            draw.rectangle(row[1], outline=row[2], width=line_width)
            draw.text((row[1][0][0], row[1][0][1]+fontYshift), row[0], font=fnt, fill=row[2])
        
        print(f'Writing {out_dir}/{prefix}_{img_name}')
        source_img.save(f'{out_dir}/{prefix}_{img_name}', "JPEG")

        
def plot_test_results(coco_test_true, coco_test_results):
    with open(coco_test_true, 'r') as j:
        ttruth = json.load(j)
    
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
        .merge(df2[['id_pred', 'score']], how='outer', on='id_pred')

    pairs = pairs.merge(df_pred[['id_pred', 'name']], how='outer', on='id_pred')

    pairs['is_correct'] = (pairs['name_x'] == pairs['name_y'])
    pairs['is_correct_class'] = (pairs['name_x'] == pairs['name_y']).where(pairs.id_pred.notnull(), np.nan)
        
    return pairs

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          write=None):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(12, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    if write:
        plt.savefig(write)
    plt.show()

def getbox(x):
    return box(x[0],x[1],x[0]+x[2],x[1]+x[3])


