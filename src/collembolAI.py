#!/usr/bin/env python3
"""
Project title:       CollembolAI
Authors:             Stephan Weißbach, Stanislav Sys, Clément Schneider
Original repository: https://github.com/stasys-hub/Collembola_AI.git
Module title:        collembolAI.py
Purpose:             Object Detection and classification for samples of
                     soil fauna invertebrates in fluid
Dependencies:        See ReadMe
Last Update:         31.01.2021
Licence:             
"""

# Imports
import traceback
import argparse
import configparser

from cocosets_utils import testresults2coco, coco2df, draw_coco_bbox, \
                           deduplicate_overlapping_preds, \
                           match_true_n_pred_box, d2_instance2dict
import cv2
import duster
from detectron2.modeling import build_model
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.structures import BoxMode
import json
import numpy as np
import ntpath
import os
import pandas as pd
import PIL
from sklearn.metrics import confusion_matrix
from third_party_utils import plot_confusion_matrix
import warnings

PIL.Image.MAX_IMAGE_PIXELS = 500000000

# Please ensure the following structure for the working directory

# working_directory ............. w
# where a set of pictures for training or automatic annotating is deposited.
# |
# |-CAI.conf ................ configuration file.
# |
# |- -|
#     |-train ............... contains the training set of pictures, with annotation in COCO format (train.json)
#     |   |-img1.jpg
#     |   |-img2.jpg
#     |   |-[...]
#     |   |-train.json
#     |
#     |-test ................ contains the testing set of pictures, with annotation in COCO format (test.json)
#     |   |-img1.jpg
#     |   |-img2.jpg
#     |   |-[...]
#     |   |-test.json
#     |
#     |-dust (optional)...... contains training set of background only pictures (no annotation needed).
#     |   |-img1.jpg
#     |   |-img2.jpg
#     |   |-[...]
#     | 
#     |-inference(optional).. contains the set of pictures to annotate with the model
#         |-img1.jpg
#         |-img2.jpg
#         |-[...]

# train: annotated images for training
# train.json: Annotations + path to images for all images stored in train (JSON/COCO format)
# test: annotated images for model evaluation
# test.json: like train.json but for test (JSON/COCO format) 
# OOP Version of CollembolAI

class collembola_ai:

    # read config from CAI.conf file 
    config = configparser.ConfigParser()
    config.read("CAI.conf")

    def __init__(self, config_path='CAI.conf', gpu_num='0'):
        """Function to initialize the CollembolaAI main class. These Parameters will be used to configure Detectron2"""
        
        
        config = configparser.ConfigParser()
        config.read(config_path)

        # set project directories
        self.project_directory = config['DEFAULT']['project_directory']
        self.model_name = config['DEFAULT']['model_name']
        self.output_directory = os.path.join(self.project_directory, self.model_name)
        self.train_directory = os.path.join(self.project_directory, "train")
        self.test_directory = os.path.join(self.project_directory, "test")
        self.dust_directory = os.path.join(self.project_directory, "dust")
        self.duster_path = os.path.join(self.project_directory, "duster")
        self.inference_directory = os.path.join(self.project_directory, config['OPTIONAL']["inference_directory"])
        
        # set model parameters
        self.num_iter = int(config['OPTIONAL']['iterations'])
        self.num_workers = int(config['OPTIONAL']['number_of_workers'])
        self.batch_size = int(config['OPTIONAL']['batch_size'])
        self.learning_rate = float(config['OPTIONAL']['learning_rate'])

        with open(os.path.join(self.train_directory, "train.json"), 'r') as js:
            self.num_classes = len(json.load(js)['categories'])

        print('Found {} classes in the training annotation file'.format(self.num_classes))
        self.threshold = float(config['OPTIONAL']['detection_threshold'])
        self.model_zoo_config = config['OPTIONAL']['model_zoo_config']

        # set gpu device to use
        self.gpu_num = int(gpu_num)
        self.trainer = None        

    def print_model_values(self):
        """This function will print all model parameters which can be set by the user. It is useful if you have path problems.
        Hint: On Windows you will probably have to adjust your path because of backslashes"""

        print("# --------------- Model Parameters ---------------- #\n")
        print(f'Variable           \tValue\n')
        print(f'Project Dir:        \t{self.project_directory}')
        print(f'Output Dir:         \t{self.output_directory}')
        print(f'Model Zoo:          \t{self.model_zoo_config}')
        print(f'Number iterations:  \t{self.num_iter}')
        print(f'Number of workers:  \t{self.num_workers}')
        print(f'Batch Size:         \t{self.batch_size}')
        print(f'Learning Rate:      \t{self.learning_rate}')
        print(f'Number of classes:  \t{self.num_classes}')
        print(f'GPU device number:  \t{self.gpu_num}')
        print(f'Treshhold:          \t{self.threshold}')
        print("\n# ------------------------------------------------- #")


    def load_train_test(self):
        """This function loads the train.json file and registers your training data using the \"register_coco_instances\" function of Detectron2
        IMPORTANT: Currently it is necessary to use this function before performing inference with a trained model """

        try:
            # read train.json file

            with open(os.path.join(self.train_directory, "train.json")) as f:
                imgs_anns = json.load(f)

            # register custom datasets in COCO format
            #                       "my_dataset", "metadata", "json_annotation.json", "path/to/image/dir
            register_coco_instances("train", {}, os.path.join(self.train_directory, "train.json"), "train")
            register_coco_instances("test", {}, os.path.join(self.test_directory, "test.json"), "test")
            dataset_dicts = DatasetCatalog.get("train")
            dataset_metadata = MetadataCatalog.get("train")

        except:
            print("ERROR!\nUnable to load model configurations!\nPlease check your input and use \"print_model_values\" for debugging ")
        
    def describe_train_test(self):
            print('Outputing some infos about the train and test dataset')
            with open(os.path.join(self.test_directory, "test.json"), 'r') as j:
                ttruth =  json.load(j)
                df_ttruth = coco2df(ttruth)
                df_ttruth['id_true'] = df_ttruth['id']

    
            with open(os.path.join(self.train_directory, "train.json"), 'r') as j:
                train =  json.load(j)
                df_train = coco2df(train)
                df_train['id_train'] = df_train['id']   

            print('Abundance of each species in the train and test pictures\n')
            tt_abundances = df_train.name.value_counts().to_frame().join(df_ttruth.name.value_counts(), lsuffix='_train', rsuffix='_test')
            tt_abundances.columns = ['Train', 'Test']
            print(tt_abundances.to_markdown())
            tt_abundances.to_csv(os.path.join(self.project_directory, "train_test_species_abundance.tsv"), sep='\t')

            print('\n\nIndividual average area per species\n')
            sum_abundance = tt_abundances.sum(axis=1)
            sum_abundance.name='abundance'
            species_stats = pd.concat([df_train.groupby('name').sum()['area'].to_frame().reset_index(),
            df_ttruth.groupby('name').sum()['area'].to_frame().reset_index()]).groupby('name').sum().join(sum_abundance)
            species_stats['avg_area'] = round(species_stats['area'] / species_stats['abundance']).astype('int')

            print(species_stats['avg_area'].to_markdown())
            species_stats['avg_area'].to_csv(os.path.join(self.project_directory, "species_avg_individual_area.tsv"), sep='\t')

    def start_training(self):
        '''This function will configure Detectron with your input Parameters and start the Training.
        HINT: If you want to check your Parameters before training use "print_model_values"'''

        # load a model from the modelzoo and initialize model weights and set our model params
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(self.model_zoo_config))
        cfg.DATASETS.TRAIN = ("train",)
        cfg.DATASETS.TEST = ()
        cfg.DATALOADER.NUM_WORKERS = self.num_workers
        cfg.OUTPUT_DIR =  self.output_directory
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.model_zoo_config)
        cfg.SOLVER.IMS_PER_BATCH = self.batch_size
        cfg.SOLVER.BASE_LR = self.learning_rate
        cfg.SOLVER.MAX_ITER = self.num_iter
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
        cfg.MODEL.RETINANET.NUM_CLASSES = self.num_classes
        cfg.nms = True
        cfg.MODEL.DEVICE = self.gpu_num
        # This will start the Trainer -> Runtime depends on hardware and parameters
        os.makedirs(self.output_directory, exist_ok=True)
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
        print("\n---------------Finished Training---------------")
        
        
    def start_training_duster(self, epochs=50):
        '''This function will train the duster (CNN binary classifier to recognize dust or other non animal object)'''
                
        with open(os.path.join(self.train_directory, "train.json"), 'r') as j:
            train =  json.load(j)
            df_train = coco2df(train)
            #df_train['id_train'] = df_train['id']
            
        ######  
        # Finding some dust using the trained rCNN model
        self.perform_inference_on_folder(inference_directory=self.dust_directory, imgtype = "jpg")
        
        with open(os.path.join(self.dust_directory, f'{self.model_name}/inferences.json'), 'r') as j:
            tdust = json.load(j)
                 
        df_dust = coco2df(tdust)
        df_dust['name'] = 'Dust'
    
        # Grabing some pieces of background in the train set (optional, currently no longer in use)
        # extract_random_background_subpictures(df_train, self.train_directory, f'{self.duster_path}/train/Dust', num_subpict_per_pict=200)
        
        print('Preparing the duster training and validation data')
        
        
        duster.dump_training_set(self.train_directory, self.dust_directory, self.duster_path, df_train, df_dust)
        print('Training and validating the duster')
        duster.train_duster(self.duster_path, self.train_directory, epochs=50)

        print('duster trained')
        

    def start_evaluation_on_test(self, dedup_thresh=0.15, verbose=True, dusting=False):
        '''This function will run the trained model on the test dataset (test/test.json)'''

        # RUNNING INFERENCE
        #================================================================================================
        # Creating the output directories tree
        os.makedirs(self.output_directory, exist_ok=True)
        output_test_res = os.path.join(self.output_directory, "test_results")
        os.makedirs(output_test_res, exist_ok=True)
        
        # Loading the model
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(self.model_zoo_config))
        cfg.DATASETS.TRAIN = ("train",)
        cfg.DATALOADER.NUM_WORKERS = self.num_workers
        cfg.OUTPUT_DIR =  self.output_directory
        cfg.SOLVER.IMS_PER_BATCH = self.batch_size
        cfg.SOLVER.BASE_LR = self.learning_rate
        cfg.SOLVER.MAX_ITER = self.num_iter
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
        cfg.MODEL.RETINANET.NUM_CLASSES = self.num_classes
        cfg.nms = True
        cfg.MODEL.DEVICE = self.gpu_num

        # Configuration for test mode
        print(os.path.join(self.output_directory, "model_final.pth"))
        cfg.MODEL.WEIGHTS = os.path.join(self.output_directory, "model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold
        cfg.DATASETS.TEST = ("test", )
        
        # Improving detection on crowed picture
        cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 10000
        cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 10000
        cfg.TEST.DETECTIONS_PER_IMAGE = 10000
        predictor = DefaultPredictor(cfg)
        
        # Preparing the test dataset
        dataset_dicts_test = DatasetCatalog.get("test")
        dataset_metadata_test = MetadataCatalog.get("test")
        
        # Configuring trainer and evaluator
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=True)
        evaluator = COCOEvaluator("test", cfg, False, output_dir=output_test_res)
        val_loader = build_detection_test_loader(cfg, "test")
        
        # Running the inference
        print(inference_on_dataset(trainer.model, val_loader, evaluator))
        #================================================================================================
        
        
        # REPORTING
        #================================================================================================
        print('Reporting and evaluating the inference on test set')
        print('\n\nLoading predicted labels from "coco_instances_results.json"')
        
        # Loading the predictions in a DataFrame, deduplicating overlaping predictions
        tpred = testresults2coco(self.test_directory, self.output_directory, write=True)
        df_pred = deduplicate_overlapping_preds(coco2df(tpred), dedup_thresh)

        # Dusting (identifying and removing False Positive ('dust'))
        if dusting:
            
                # Extracting subpictures from the predictions
            print('Extracting and organizing the subpictures for dusting')
            
            def wipe_dir(path):
                if os.path.exists(path) and os.path.isdir(path):
                    shutil.rmtree(path)
        
            wipe_dir(os.path.join(self.duster_path,'to_predict'))
            
        
            os.makedirs(os.path.join(self.duster_path,'to_predict/All'), exist_ok=True)
        
            for file in df_pred.file_name.unique():
                im = Image.open(self.test_directory + '/' + file)
        
                for name in df_pred['name'].unique():
                    os.makedirs(os.path.join(self.duster_path, f'to_predict/{name}'), exist_ok=True)
               
                    for raw in df_pred[(df_pred['file_name']==file) & (df_pred['name']==name)][['box', 'id', 'name']].values:
                        im.crop(raw[0].bounds).save(f'{self.duster_path}/to_predict/All/{raw[1]}.jpg','JPEG')
                        im.crop(raw[0].bounds).save(f'{self.duster_path}/to_predict/{raw[2]}/{raw[1]}.jpg','JPEG')
                im.close()
            
            
            list_classes = list(df_pred.name.unique())
            duster.load_duster_and_classify(self.duster_path, list_classes)
            dust = pd.read_csv(f'{self.duster_path}/dust.csv')
            df_pred = df_pred.merge(dust, on='id')
            df_pred = df_pred[df_pred['dust'] != 'Dust']
        
        
        # Loading train set and test set in DataFrame
        with open(os.path.join(self.test_directory, "test.json"), 'r') as j:
                ttruth =  json.load(j)
                df_ttruth = coco2df(ttruth)
                df_ttruth['id_true'] = df_ttruth['id']

        with open(os.path.join(self.train_directory, "train.json"), 'r') as j:
                train =  json.load(j)
                df_train = coco2df(train)
                df_train['id_train'] = df_train['id']
        
        # Computing representation (abundance and area) of each classes in the train and test dataset
        #------------------------------------------------------------------------------------------------
        tt_abundances = df_train.name.value_counts().to_frame().join(df_ttruth.name.value_counts(), lsuffix='_train', rsuffix='_test')
        tt_abundances.columns = ['Train', 'Test']         
        tt_abundances = tt_abundances.join(df_pred.name.value_counts())\
                                    .join(df_ttruth.groupby('name').sum()['area'])\
                                    .join(df_pred.groupby('name').sum()['area'], rsuffix="pred")
        tt_abundances.columns = ['Train', 'Test True', 'Test Pred', 'Test True Area', 'Test Pred Area']
        tt_abundances['Perc Pred True'] = tt_abundances['Test Pred Area'] / tt_abundances['Test True Area'] * 100
        tt_abundances['Test True Contribution To Total Area'] =  tt_abundances['Test True Area'] / tt_abundances['Test True Area'].sum() * 100
        tt_abundances['Test Pred Contribution To Total Area'] =  tt_abundances['Test Pred Area'] / tt_abundances['Test Pred Area'].sum() * 100
        tt_abundances.to_csv(os.path.join(self.output_directory, "test_results/species_abundance_n_area.tsv"), sep='\t')
        #------------------------------------------------------------------------------------------------
        
        # Matching the predicted annotations with the true annotations
        pairs = match_true_n_pred_box(df_ttruth, df_pred, IoU_threshold=0.4)

        # Computing detection rate, classification accuracy, false positive rate
        #------------------------------------------------------------------------------------------------
        total_true_labels = pairs.id_true.notnull().sum()
        true_labels_without_matching_preds = pairs.id_pred.isnull().sum()
        perc_detected_animals = 100 - (true_labels_without_matching_preds / total_true_labels * 100)
        perc_correct_class = pairs['is_correct_class'].sum() / pairs.dropna().shape[0] * 100

        if verbose:
            print(f'The test set represents a total of {total_true_labels} specimens.')
            print(f'The model produced {len(tpred["annotations"])} prediction, of which {df_pred.shape[0]} remains after deduplication' +
                   ' and removal of oversized bounding boxes.')
            print(f'{total_true_labels - true_labels_without_matching_preds} ({round(perc_detected_animals, 1)}% of the total) ' + 
                   'of the actual specimens were correcly detected.' +
                  f' Of those detected specimens, {int(pairs["is_correct_class"].sum())} (= {round(perc_correct_class, 1)}%) where assigned to the correct species.')

        
        # Tagging the false positives in df_pred
        df_pred = df_pred.merge(pairs[['id_pred', 'id_true']], how='left', on='id_pred')
        df_pred['is_false_positive'] = True
        df_pred['is_false_positive'] = df_pred['is_false_positive'].where(df_pred['id_true'].isnull(), False)
        
        # Adding inference outcomes on the true labels, df_ttruth
        df_ttruth = df_ttruth.merge(pairs[pairs['name_true'].notnull()][['id_true', 'score', 'name_pred', 'is_correct_class']], on='id_true')
        df_ttruth['is_detected'] = df_ttruth['is_correct_class'].where(df_ttruth['is_correct_class'].isnull(), 1).fillna(0)

        if verbose:
            print(f'Of the predicted labels, {df_pred["is_false_positive"].sum()} '+
                  f'(={round(df_pred["is_false_positive"].sum() / df_pred.shape[0] * 100,1)}%) '+
                   'where false positive (background, not related to a real specimen)')
        #------------------------------------------------------------------------------------------------
              
        # Drawing the predicted annotations on the pictures
        #------------------------------------------------------------------------------------------------
        print('\n\nDrawing the predicted annotations of the test pictures to support visual verification')
        print('Do not use for testing or for training ! =)')
        draw_coco_bbox(df_pred, os.path.join(self.output_directory, "test_results"), self.test_directory, 
                       prefix="predicted", line_width=10, fontsize = 150, fontYshift = -125)
        #------------------------------------------------------------------------------------------------

        # Plotting the confusion matrices
        #------------------------------------------------------------------------------------------------
        # 1. CM including only the detected true label
        mcm = confusion_matrix(pairs.dropna().name_true, pairs.dropna().name_pred.fillna('NaN'), labels = pairs.dropna().name_true.unique())
        plot_confusion_matrix(mcm, pairs.dropna().name_true.unique(), normalize=False,
                write=os.path.join(self.output_directory, "test_results/cm_onlydetected.png"))
        
        # 2. CM including only the detected true label, normalized
        # Note: the normalized matrix option is bugged in the plot_confusion_matrix function from sklearn
        # Thus I normalize the matrix here before plotting and don't use the option
        mcm = mcm.astype('float') / mcm.sum(axis=1)[:, np.newaxis] * 100
        mcm = mcm.round(1)
        plot_confusion_matrix(mcm, pairs.dropna().name_true.unique(), normalize=False, 
                write=os.path.join(self.output_directory, "test_results/cm_norm_onlydetected.png"))
        
        # 3. CM including only the undetected true label (Nan)
        mcm = confusion_matrix(pairs.name_true.fillna('NaN'), pairs.name_pred.fillna('NaN'), labels = pairs.fillna('NaN').name_true.unique())
        plot_confusion_matrix(mcm, np.append(pairs.name_true.unique(), 'NaN'), normalize=False,
                write=os.path.join(self.output_directory, "test_results/cm_inclNaN.png"))
        
        # 4. CM including only the undetected true label (Nan), normalized
        mcm = mcm.astype('float') / mcm.sum(axis=1)[:, np.newaxis] * 100
        mcm = np.nan_to_num(mcm.round(1))
        plot_confusion_matrix(mcm, pairs.fillna('NaN').name_true.unique(), normalize=False, 
                write=os.path.join(self.output_directory, "test_results/cm_norm_inclNaN.png"))     

        print("\n---------------Finished Evaluation---------------")
        
        #================================================================================================

    def perform_inference_on_folder(self, inference_directory = None, imgtype = "jpg"):
        '''This function can be used to perform inference on the unannotated data you want to classify.
           IMPORTANT: You still have to load a model using "load_train_test"'''
        
        if not inference_directory:
            inference_directory = self.inference_directory

        try:
            # reload the model
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file(self.model_zoo_config))
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
            cfg.nms = True

            #config for test mode
            cfg.MODEL.WEIGHTS = os.path.join(self.output_directory, "model_final.pth")
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold
            cfg.DATASETS.TEST = ("test", )
            cfg.MODEL.DEVICE = self.gpu_num 
            #Improving detection on crowed picture ?
            cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 10000
            cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 10000
            cfg.TEST.DETECTIONS_PER_IMAGE = 10000
            
            predictor = DefaultPredictor(cfg)

            #prepare test dataset for metadata
            dataset_dicts_test = DatasetCatalog.get("test")
            dataset_metadata_test = MetadataCatalog.get("test")
        except Exception as e:
            print(e)
            print("Something went wrong while loading the model configuration. Please Check your path\'")
            raise
 
        inference_out = os.path.join(inference_directory, self.model_name)          
        n_files = len(([f for f in os.listdir(inference_directory) if f.endswith(imgtype)]))
        counter = 1

        print(f"Starting inference ...\nNumber of Files:\t{n_files}\nImage extension:\t{imgtype}")
        print("\n# ------------------------------------------------- #\n")

        try:
            # I added this because Detectron2 uses an deprecated overload -> throws warning
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                os.makedirs(inference_out, exist_ok=True)
                results_coco={'images': [], 'annotations': []}
                for i in os.listdir(inference_directory):
                    if i.endswith(imgtype):
                        file_path = os.path.join(inference_directory, i)
                        print(f"processing [{counter}/{n_files}]", end='\r')
                        im = cv2.imread(file_path)
                        outputs = predictor(im)
                        v = Visualizer(im[:, :, ::-1], metadata=dataset_metadata_test, scale=1.)
                        instances = outputs["instances"].to("cpu")
                        coco = d2_instance2dict(instances, counter, ntpath.basename(i))
                        v = v.draw_instance_predictions(instances)
                        result = v.get_image()[:, :, ::-1]
                        output_name = f"{inference_out}/annotated_{i}"
                        write_res = cv2.imwrite(output_name, result)
                        counter += 1

                        results_coco['images'] = results_coco['images'] + coco['images']
                        results_coco['annotations'] = results_coco['annotations'] + coco['annotations']
                        
                annotation_id = 0
                for a in results_coco['annotations']:
                        a['id'] = annotation_id
                        annotation_id += 1
                results_coco['type'] = 'instances'
                results_coco['licenses'] = ''
                results_coco['info'] = ''

                with open(os.path.join(self.train_directory, "train.json"), 'r') as j:
                    train = json.load(j)
                results_coco['categories'] = train['categories']
                        
                        
                with open(os.path.join(inference_out, "inferences.json"), 'w') as j:
                    json.dump(results_coco, j, indent=4)
                        

                
        except Exception as e:
            print(e)
            print("Something went wrong while performing inference on your data. Please check your path directory structure\nUse \"print_model_values\" for debugging")



def main():
    parser=argparse.ArgumentParser()

    parser.add_argument('config_file', type=str, 
            help='''Path of the configuration file (default: "./CAI.conf")''')
    
    parser.add_argument('-t', '--train',action='store_true',
            help='''(re-)Train a model using the train set of pictures (default: skip)''')
    
    parser.add_argument('-d', '--train_duster',action='store_true',
            help='''(re-)Train the CNN "duster" using the train set of pictures (require a trained rCNN first, default: skip)''')
              
    parser.add_argument('-e', '--evaluate',action='store_true',
            help='''Evaluate the model using the test set of pictures (default: skip)''')
    
    parser.add_argument('-a', '--annotate',action='store_true',
            help='''Annotate the inference set of pictures (default: skip)''')

    parser.add_argument('-s', '--sets_description',action='store_true',
            help='''Output some descriptions elements for the train and test set in the project directory''')
    
    parser.add_argument('--visible_gpu', type=str, default="0",
            help='''List of visible gpu to CUDA (default: "0", example: "0,1")''')
    
    parser.add_argument('--gpu_num', type=int, default=0,
            help='''Set the gpu device number to use (default: 0)''')

    args=parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpu


    # Example: Run CollembolAI with your defined parameters
    # Define your model parameters
    My_Model = collembola_ai(config_path = args.config_file, gpu_num = args.gpu_num)
              
    # print the model parameters
    My_Model.print_model_values()
              
    # register the training and My_Model.sets in detectron2
    My_Model.load_train_test()

    if args.sets_description:
        My_Model.describe_train_test()

    if args.train:
        # start training 
        My_Model.start_training()
    else:
        print("Skipping training")
              
    if args.train_duster:
        # start training 
        My_Model.start_training_duster(epochs=50)
    else:
        print("Skipping duster training")

    if args.evaluate:
        # start evaluation on My_Model.set
        My_Model.start_evaluation_on_test(dusting=True)
    else:
        print("Skipping evaluation")

    if args.annotate:
        # Run inference with your trained model on unlabeled data       
        My_Model.perform_inference_on_folder(imgtype="jpg")
    else:
        print("Nothing to annotate")


if __name__ == "__main__":
    main()
