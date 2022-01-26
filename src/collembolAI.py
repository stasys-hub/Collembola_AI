#!/usr/bin/env python3
"""
Project title:       CollembolAI
Authors:             Stephan Weißbach, Stanislav Sys, Clément Schneider
Original repository: https://github.com/stasys-hub/Collembola_AI.git
Module title:        collembolAI.py
Purpose:             Object Detection and classification for samples of
                     soil fauna invertebrates in fluid
Dependencies:        See ReadMe
Last Update:         11.01.2021
Licence:             
"""

# Imports
import traceback
import argparse
import configparser

from cocosets_utils import testresults2coco, coco2df, draw_coco_bbox, \
                           deduplicate_overlapping_preds, \
                           match_true_n_pred_box
import cv2
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
import os
import pandas as pd
import PIL
from sklearn.metrics import confusion_matrix
from third_party_utils import plot_confusion_matrix
import warnings

PIL.Image.MAX_IMAGE_PIXELS = 500000000

# please ensure the following structure:
# project_directory ............. working directory, where a set of pictures for training or automatic annotating is deposited.
# |
# |
# |-collembolaAI.py 
# |-CAI.conf .................... configuration file.
# |
# |- -|
#     |-train ................... contains the training set of pictures, with annotation in COCO format (train.json)
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
#     |-inference(optional) ....  contains the set of pictures to annotate with the model
#         |-img1.jpg
#         |-img2.jpg
#         |-[...]


# train: Annotated Images for training
# train.json: Annotations + Path to Images for all images stored in train
# test: Annotated Images for model evaluation (Taken from train)
# test.json: Like train.json but for test

# OOP Version of CollembolaAI

class collembola_ai:

    # read config from CAI.conf file 
    config = configparser.ConfigParser()
    config.read("CAI.conf")

    # def __init__(self, workingdir: str, outputdir: str, n_iterations: int = 8000, work_num: int = 2, my_batch_size: int = 5, learning_rate: float = 0.00025, number_classes:int = 10, treshhold: float = 0.55):
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
        self.inference_directory = os.path.join(self.project_directory, config['OPTIONAL']["inference_directory"])
        
        # set model parameters
        self.num_iter = int(config['OPTIONAL']['iterations'])
        self.num_workers = int(config['OPTIONAL']['number_of_workers'])
        self.batch_size = int(config['OPTIONAL']['batch_size'])
        self.learning_rate = float(config['OPTIONAL']['learning_rate'])

        with open(os.path.join(self.train_directory, "train.json"), 'r') as js:
            self.num_classes = len(json.load(js)['categories'])

        print('Found {} classes in the training annotation file'.format(self.num_classes))
        self.threshold = float(config['OPTIONAL']['detection_treshold'])
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

    def start_evaluation_on_test(self):
        """This function will start the testing the model on test_set1 and test_set2"""

        # reload the model
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

        #config for test mode
        print(os.path.join(self.output_directory, "model_final.pth"))
        cfg.MODEL.WEIGHTS = os.path.join(self.output_directory, "model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold
        cfg.DATASETS.TEST = ("test", )
        
        #Improving detection on crowed picture ?
        cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 10000
        cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 10000
        cfg.TEST.DETECTIONS_PER_IMAGE = 10000
        predictor = DefaultPredictor(cfg)
        #prepare test dataset
        dataset_dicts_test = DatasetCatalog.get("test")
        dataset_metadata_test = MetadataCatalog.get("test")

        #annotate all pictures of testset1 and testset2 with the predictions of the trained model
        os.makedirs(self.output_directory, exist_ok=True)
        output_test_res = os.path.join(self.output_directory, "test_results")
        os.makedirs(output_test_res, exist_ok=True)
        #output_testset2 = os.path.join(self.output_directory, "testset2")
        #os.makedirs(output_testset2, exist_ok=True)

        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=True)

        evaluator = COCOEvaluator("test", cfg, False, output_dir=output_test_res)
        val_loader = build_detection_test_loader(cfg, "test")
        print(inference_on_dataset(trainer.model, val_loader, evaluator))
        
        # From Clem: I deactivate the following block, its function is replaced by the five above lines
        # I also made a custom function for vizualisation which may not be as streamlined
        # as the detectron visualizer, but so far provide a more readble output.
        
        #try:
        #    i = 0
        #    for d in dataset_dicts_test:
        #         #create variable with output name
        #        output_name = output_test_res + "/annotated_" + str(i) + ".jpg"
        #        img = cv2.imread(d["file_name"])
        #        print(f"Processing: \t{output_name}")
        #        #make prediction
        #        outputs = predictor(img)
        #        #draw prediction
        #        visualizer = Visualizer(img[:, :, ::-1], metadata=dataset_metadata_test, scale=1.)
        #        instances = outputs["instances"].to("cpu")
        #        vis = visualizer.draw_instance_predictions(instances)
        #        result = vis.get_image()[:, :, ::-1]
        #        #write image
        #        write_res = cv2.imwrite(output_name, result)
        #        i += 1
        #except:
        #    traceback.print_exc()
        #    print("Something went wrong while evaluating the \"test\" set. Please check your path directory structure\nUse \"print_model_values\" for debugging")

        #model = build_model(cfg)

        #evaluator = COCOEvaluator("test", cfg, False, output_dir=output_test_res)
        #val_loader = build_detection_test_loader(cfg, "test")
        #print(inference_on_dataset(model, val_loader, evaluator))
        #output_testset2 = os.path.join(self.output_directory, "testset2")
        #os.makedirs(output_testset2, exist_ok=True)
        print('Report on evaluation')
        
        print('\n\nLoading predicted labels from "coco_instances_results.json", ,)
        tpred = testresults2coco(self.test_directory, self.output_directory, write=True)

        print('\n\nLoading predictions and deduplicating overlaping predictions')
        df_pred = deduplicate_overlapping_preds(coco2df(tpred), 0.4)
        
        with open(os.path.join(self.test_directory, "test.json"), 'r') as j:
                ttruth =  json.load(j)
                df_ttruth = coco2df(ttruth)
                df_ttruth['id_true'] = df_ttruth['id']

        with open(os.path.join(self.train_directory, "train.json"), 'r') as j:
                train =  json.load(j)
                df_train = coco2df(train)
                df_train['id_train'] = df_train['id']

        tt_abundances = df_train.name.value_counts().to_frame().join(df_ttruth.name.value_counts(), lsuffix='_train', rsuffix='_test')
        tt_abundances.columns = ['Train', 'Test']         

        print('\n\nAbundance and area of each species in the train and test pictures (true and predicted)\n')
        tt_abundances = tt_abundances.join(df_pred.name.value_counts())\
                                    .join(df_ttruth.groupby('name').sum()['area'])\
                                    .join(df_pred.groupby('name').sum()['area'], rsuffix="pred")
        tt_abundances.columns = ['Train', 'Test True', 'Test Pred', 'Test True Area', 'Test Pred Area']
        tt_abundances['Perc Pred True'] = tt_abundances['Test Pred Area'] / tt_abundances['Test True Area'] * 100
        tt_abundances['Test True Contribution To Total Area'] =  tt_abundances['Test True Area'] / tt_abundances['Test True Area'].sum() * 100
        tt_abundances['Test Pred Contribution To Total Area'] =  tt_abundances['Test Pred Area'] / tt_abundances['Test Pred Area'].sum() * 100
        print(tt_abundances.to_markdown())
        print(self.output_directory)
        tt_abundances.to_csv(os.path.join(self.output_directory, "test_results/species_abundance_n_area.tsv"), sep='\t')
        pairs = match_true_n_pred_box(df_ttruth, df_pred, IoU_threshold=0.4)
              
              
        total_true_labels = pairs.id_true.notnull().sum()
        true_labels_without_matching_preds = pairs.id_pred.isnull().sum()
        perc_detected_animals = 100 - (true_labels_without_matching_preds / total_true_labels * 100)
        perc_correct_class = pairs['is_correct_class'].sum() / pairs.dropna().shape[0] * 100

        #print(f'\n')
        print(f'The test set represents a total of {total_true_labels} specimens.')
        print(f'The model produced {len(tpred["annotations"])} prediction, of which {df_pred.shape[0]} remains after deduplication' +
               ' and removal of oversized bounding boxes.')
        print(f'{total_true_labels - true_labels_without_matching_preds} ({round(perc_detected_animals, 1)}% of the total) ' + 
               'of the actual specimens were correcly detected.' +
               f' Of those detected specimens, {int(pairs["is_correct_class"].sum())} (= {round(perc_correct_class, 1)}%) where assigned to the correct species.')

        # Tagging False positive on df_pred
        df_pred = df_pred.merge(pairs[['id_pred', 'id_true']], how='left', on='id_pred')
        df_pred['is_false_positive'] = True
        df_pred['is_false_positive'] = df_pred['is_false_positive'].where(df_pred['id_true'].isnull(), False)

        print(f'Of the predicted labels, {df_pred["is_false_positive"].sum()} '+
        f'(={round(df_pred["is_false_positive"].sum() / df_pred.shape[0] * 100,1)}%) '+
         'where false positive (background, not related to a real specimen)')

        print('\n\nDrawing the predicted annotations of the test pictures to support visual verification')
        print('Do not use for testing or for training ! =)')
        draw_coco_bbox(df_pred, os.path.join(self.output_directory, "test_results"), self.test_directory, prefix="predicted", line_width=10, fontsize = 150, fontYshift = -125)

        mcm = confusion_matrix(pairs.dropna().name_x, pairs.dropna().name_y.fillna('NaN'), labels = pairs.dropna().name_x.unique())
        plot_confusion_matrix(mcm, pairs.dropna().name_x.unique(), normalize=False,
                 write=os.path.join(self.output_directory, "test_results/cm_onlydetected.png"))

        #The normalized matrix output is bugged in the plot_confusion_matrix function from sklearn, thus I normalize the 
        # matrix here before plotting
        mcm = mcm.astype('float') / mcm.sum(axis=1)[:, np.newaxis] * 100
        mcm = mcm.round(1)
        plot_confusion_matrix(mcm, pairs.dropna().name_x.unique(), normalize=False, 
                   write=os.path.join(self.output_directory, "test_results/cm_norm_onlydetected.png"))

        mcm = confusion_matrix(pairs.name_x.fillna('NaN'), pairs.name_y.fillna('NaN'), labels = pairs.fillna('NaN').name_x.unique())
        plot_confusion_matrix(mcm, np.append(pairs.name_x.unique(), 'NaN'), normalize=False,
                   write=os.path.join(self.output_directory, "test_results/cm_inclNaN.png"))
        mcm = mcm.astype('float') / mcm.sum(axis=1)[:, np.newaxis] * 100
        mcm = np.nan_to_num(mcm.round(1))                                                                                 
        plot_confusion_matrix(mcm, pairs.fillna('NaN').name_x.unique(), normalize=False, 
                         write=os.path.join(self.output_directory, "test_results/cm_norm_inclNaN.png"))     

        print("\n---------------Finished Evaluation---------------")

    def perfom_inference_on_folder(self, imgtype = "jpg"):
        '''This function can be used to test a trained model with a set of images or to perform inference on data you want to classify.
        IMPORTANT: You still have to load a model using "load_train_test"'''
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
 
        inference_out = os.path.join(self.inference_directory, self.model_name)          
        n_files = len(([f for f in os.listdir(self.inference_directory) if f.endswith(imgtype)]))
        counter = 1

        print(f"Starting inference ...\nNumber of Files:\t{n_files}\nImage extension:\t{imgtype}")
        print("\n# ------------------------------------------------- #\n")

        try:
            # I added this because Detectron2 uses an deprecated overload -> throws warning
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                os.makedirs(inference_out, exist_ok=True)
                for i in os.listdir(self.inference_directory):
                    if i.endswith(imgtype):
                        file_path = os.path.join(self.inference_directory, i)
                        print(f"processing [{counter}/{n_files}]", end='\r')
                        im = cv2.imread(file_path)
                        outputs = predictor(im)
                        v = Visualizer(im[:, :, ::-1], metadata=dataset_metadata_test, scale=1.)
                        instances = outputs["instances"].to("cpu")
                        v = v.draw_instance_predictions(instances)
                        result = v.get_image()[:, :, ::-1]
                        output_name = f"{inference_out}/annotated_{i}"
                        write_res = cv2.imwrite(output_name, result)
                        counter += 1
        except:
            print("Something went wrong while performing inference on your data. Please check your path directory structure\nUse \"print_model_values\" for debugging")



def main():
    parser=argparse.ArgumentParser()

    parser.add_argument('config_file', type=str, 
            help='''Path of the configuration file (default: "./CAI.conf")''')
    
    parser.add_argument('-t', '--train',action='store_true',
            help='''(re-)Train a model using the train set of pictures (default: skip)''')
    
    parser.add_argument('-e', '--evaluate',action='store_true',
            help='''Evaluate the model using the test set of pictures (default: skip)''')
    
    parser.add_argument('-a', '--annotate',action='store_true',
            help='''Annotate the inference set of pictures (default: skip)''')

    parser.add_argument('-d', '--describe_train_test',action='store_true',
            help='''Output some descriptions elements for the train and test set in the project directory''')
    
    parser.add_argument('--visible_gpu', type=str, default="0",
            help='''List of visible gpu to CUDA (default: "0", example: "0,1")''')
    
    parser.add_argument('--gpu_num', type=int, default=0,
            help='''Set the gpu device number to use (default: 0)''')

    args=parser.parse_args()

        
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpu


    # Example: Run Collembola_AI with your defined parameters
    # Define your model parameters
    
    My_Model = collembola_ai(config_path = args.config_file, gpu_num = args.gpu_num)
    # print the model parameters
    My_Model.print_model_values()
    # register the training and My_Model.sets in detectron2
    My_Model.load_train_test()

    if args.describe_train_test:
        My_Model.describe_train_test()

    if args.train:
        # start training 
        My_Model.start_training()
    else:
        print("Skipping training")

    if args.evaluate:
        # start evaluation on My_Model.set
        My_Model.start_evaluation_on_test()
    else:
        print("Skipping evaluation")

    if args.annotate:
        # Run inference with your trained model on unlabeled data       
        #set the image type ( jpg, png, etc...)
        my_type = "jpg"
        # run the objectdetection
        My_Model.perfom_inference_on_folder(imgtype="jpg")
    else:
        print("Nothing to annotate")


if __name__ == "__main__":
    main()
