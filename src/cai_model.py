"""
Project title:       CollembolAI
Authors:             Stephan Weißbach, Stanislav Sys, Clément Schneider
Original repository: https://github.com/stasys-hub/Collembola_AI.git
Module title:        collembolAI.py
Purpose:             Object Detection and classification for samples of
                     soil fauna invertebrates in fluid
Dependencies:        See ReadMe
Last Update:         31.01.2022
Licence:             
"""

import os
import json
import configparser
import shutil

# import postprocess.duster as duster


from utils.cocoutils import (
    testresults2coco,
    coco2df,
    create_coco_json_for_inference,
    sahi_result_to_coco,
    extract_classes_from_coco_json,
)

from utils.output_inference_images import draw_coco_bbox
from postprocess.nms import non_max_supression
from evaluation_functions import process_results, get_average_precision_recall_from_cls_vc, get_mAP_from_TruthPred_df

from PIL import Image
import numpy as np


from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from sahi.predict import predict
from sahi import slicing


class collembola_ai:
    def __init__(self, config_path: str):
        """
        ARGS:
            config_path: str with absolute path to configuration file
        RETURN:
            return initialized collembola model

        Function to initialize the CollembolaAI main class. These Parameters will be used to configure Detectron2
        """

        config = configparser.ConfigParser()
        config.read(config_path)

        # set project directories
        self.project_directory = config["DEFAULT"]["project_directory"]
        self.model_name = config["DEFAULT"]["model_name"]
        self.model_directory = os.path.join(self.project_directory, self.model_name)
        self.train_directory = os.path.join(self.project_directory, "train")
        self.test_directory = os.path.join(self.project_directory, "test")
        self.model_zoo_config = config["OPTIONAL"]["model_zoo_config"]
        # set model parameters
        self.num_iter = int(config["OPTIONAL"]["iterations"])
        self.num_workers = int(config["OPTIONAL"]["number_of_workers"])
        self.batch_size = int(config["OPTIONAL"]["batch_size"])
        self.learning_rate = float(config["OPTIONAL"]["learning_rate"])
        self.threshold = float(config["OPTIONAL"]["detection_threshold"])
        # set SAHI paramters
        self.slice_height = int(config["DEFAULT"]["slice_height"])
        self.slice_width = int(config["DEFAULT"]["slice_width"])
        self.overlap_height_ratio = float(config["DEFAULT"]["overlap_height_ratio"])
        self.overlap_width_ratio = float(config["DEFAULT"]["overlap_width_ratio"])
        self.class_agnostic = config.getboolean("DEFAULT", "class_agnostic")
        with open(os.path.join(self.train_directory, "train.json"), "r") as js:
            self.num_classes = len(json.load(js)["categories"])
            print(f"Found {self.num_classes} classes in the training annotation file")
        self.nms_iou_threshold = float(
            config["OPTIONAL"]["non_maximum_supression_iou_threshold"]
        self.match_iou_thresh = float(onfig["OPTIONAL"]["true_pred_matching_iou_threshold"]
        )

    def print_model_values(self):
        """
        ARGS)


        RETURN)


        This function will print all model parameters which can be set by the user.
        It is useful if you have path problems.
        """

        ###
        # SAHI: may require some update ?
        ###

        print("# ----------------------- Model Parameters ------------------------ #\n")
        print(f"Variable           \tValue\n")
        print(f"Project Dir:        \t{self.project_directory}")
        print(f"Output Dir:         \t{self.model_directory}")
        print(f"Model Zoo:          \t{self.model_zoo_config}")
        print(f"Number iterations:  \t{self.num_iter}")
        print(f"Number of workers:  \t{self.num_workers}")
        print(f"Batch Size:         \t{self.batch_size}")
        print(f"Learning Rate:      \t{self.learning_rate}")
        print(f"Number of classes:  \t{self.num_classes}")
        print(f"GPU device number:  \t{self.gpu_num}")
        print(f"Treshhold:          \t{self.threshold}")
        print("\n# ----------------------------------------------------------------- #")

    def slice_trainset(self):
        """
        This function takes an unsliced dataset and produces an scliced version.
        The folder should contain a coco annotation file. The original folder will
        be renamed to input_unscliced and the folder with the sliced images will
        get the name of the train directory.
        """
        if not os.path.isdir(self.train_directory):
            raise FileNotFoundError(
                f"Did not find train directory in project folder. Please provide train directory with path {self.train_directory}."
            )
        if not os.path.isfile(os.path.join(self.train_directory, "train.json")):
            raise FileNotFoundError(
                "Please provide a coco annotation file 'train.json'."
            )
        unsliced_train_directory = self.train_directory + "_unscliced"
        shutil.move(self.train_directory, unsliced_train_directory)
        os.mkdir(self.train_directory)
        slicing.slice_coco(
            coco_annotation_file_path=os.path.join(
                unsliced_train_directory, "train.json"
            ),
            image_dir=unsliced_train_directory,
            output_coco_annotation_file_name=os.path.join(
                self.train_directory, "train.json"
            ),
            output_dir=self.train_directory,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_height_ratio,
            overlap_width_ratio=self.overlap_width_ratio,
        )

    def load_train_test(self):
        """
        This function loads the train.json file and registers your training data using the \"register_coco_instances\" function of Detectron2
        IMPORTANT: Currently it is necessary to use this function before performing inference with a trained model
        """
        # register custom datasets in COCO format
        #                       "my_dataset", "metadata", "json_annotation.json", "path/to/image/dir
        register_coco_instances(
            "train", {}, os.path.join(self.train_directory, "train.json"), "train"
        )

    def start_training(self):
        '''This function will configure Detectron with your input Parameters and start the Training.
        HINT: If you want to check your Parameters before training use "print_model_values"'''

        ##
        ## UPDATE WITH THE TRAINING SCRIPT USING SAHI
        ##
        # check if the train images are sliced
        Image.MAX_IMAGE_PIXELS = None
        with open(os.path.join(self.train_directory, "train.json"), "r") as train_json:
            train_coco = json.load(train_json)
        im = Image.open(
            os.path.join(self.train_directory, train_coco["images"][0]["file_name"])
        )
        width, height = im.size
        if width > self.slice_width or height > self.slice_height:
            print("----------------------------------")
            print("----------------------------------")
            print("Loaded one image from train directory")
            print(train_coco["images"][0]["file_name"])
            print(f"width: {width}px \t height: {height}px")
            print("Config file specifies")
            print(f"width: {self.slice_width}px \t height: {self.slice_height}px")
            print("----------------------------------")
            print("----------------------------------")
            raise ValueError(
                "The size of the images in the train directory is larger than specified in the config file. Please run slicing first (flag '--slice') and ensure that the train directory is named 'train'."
            )
        # load a model from the modelzoo and initialize model weights and set our model params
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(self.model_zoo_config))
        cfg.DATASETS.TRAIN = ("train",)
        cfg.DATASETS.TEST = ()
        cfg.DATALOADER.NUM_WORKERS = self.num_workers
        cfg.OUTPUT_DIR = self.model_directory
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.model_zoo_config)
        cfg.SOLVER.IMS_PER_BATCH = self.batch_size
        cfg.SOLVER.BASE_LR = self.learning_rate
        cfg.SOLVER.MAX_ITER = self.num_iter
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
        cfg.MODEL.RETINANET.NUM_CLASSES = self.num_classes
        cfg.nms = True
        cfg.MODEL.DEVICE = self.gpu_num
        # This will start the Trainer -> Runtime depends on hardware and parameters
        os.makedirs(self.model_directory, exist_ok=True)
        with open(
            os.path.join(self.model_directory, "model_parameters.yaml"), "w"
        ) as outfile:
            configuration = cfg.dump().split("\n")
            # remove unneeded configurations that are not needed for SAHI (and lead to errors)
            for idx, parameter in enumerate(configuration):
                if "DEVICE" in parameter or "nms" in parameter:
                    configuration.pop(idx)
            configuration = "\n".join(configuration)
            outfile.write(configuration)
        extract_classes_from_coco_json(
            os.path.join(self.train_directory, "train.json"),
            os.path.join(self.model_directory, "classes.json"),
        )
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
        cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
        trainer.train()
        print("\n---------------Finished Training---------------")

    def start_evaluation_on_test(self, verbose=True):
        """This function will run the trained model on the test dataset (test/test.json)"""

        ##
        ## UPDATE WITH THE TEST INFERENCE SCRIPT USING SAHI
        ##

        # RUNNING INFERENCE
        # ================================================================================================
        if not os.path.isfile(os.path.join(self.test_directory, "test.json")):
            raise FileNotFoundError(
                f"Did not find the test.json path in test diretory: {self.test_directory}."
            )
        if not os.path.isfile(os.path.join(self.model_directory, "classes.json")):
            extract_classes_from_coco_json(
                os.path.join(self.test_directory, "test.json"),
                os.path.join(self.model_directory, "classes.json"),
            )
        with open(os.path.join(self.model_directory, "classes.json"), "r") as classes:
            labels = json.load(classes)

        # do batch inference on test set
        # returns relative path to resulting JSON
        export_dir = predict(
            model_type="detectron2",
            slice_width=self.slice_width,
            slice_height=self.slice_height,
            overlap_height_ratio=self.overlap_height_ratio,
            overlap_width_ratio=self.overlap_width_ratio,
            source=self.test_directory,
            model_path=os.path.join(self.model_directory, "model_final.pth"),
            model_config_path=os.path.join(
                self.model_directory, "model_parameters.yaml"
            ),
            no_standard_prediction=True,
            export_visual=False,
            return_dict=True,
            model_category_mapping=labels,
            model_confidence_threshold=self.threshold,
            dataset_json_path=os.path.join(self.test_directory, "test.json"),
            model_device="cuda",
        )["export_dir"]

        # RUNNING EVALUATION
        # ================================================================================================
        print("Reporting and evaluating the inference on test set", end="\n\n\n")
        output_directory = os.path.join(
                        self.project_directory, "test_results")
        pairs, cls_value_counts, df_ttruth, df_pred = process_results(self.test_directory, 
                                            output_directory, self.train_directory, nms_IoU=self.nms_iou_threshold,
                                            match_thresh=self.match_iou_thresh, score_thresh=self.threshold,
                                            verbose = True, draw_n_plot = True)

        metrics = [round(r, 3) for r in get_average_precision_recall_from_cls_vc(cls_value_counts)]
        print(f'Micro averaged metrics - precision: {metrics[0]}, recall: {metrics[1]}')
        print(f'Macro averaged metrics - precision: {metrics[2]}, recall: {metrics[3]}')
        
        print(f'Computing PASCAL VOC mAP@0.5')
        pairs, cls_value_counts, df_ttruth, df_pred = process_results(self.test_directory,
                                            output_directory, self.train_directory, nms_IoU=self.nms_iou_threshold,
                                            match_thresh=0.5, score_thresh=0,
                                            verbose = False, draw_n_plot = False, write_outputs=False)
        print(f'PASCAL VOC mAP@0.5:', round(get_mAP_from_TruthPred_df(pairs), 3))
     
        print("\n---------------Finished Evaluation---------------")

        # ================================================================================================

    def perform_inference_on_folder(
        self, inference_source_directory: str, inference_result_directory: str
    ):
        """
        This function can be used to perform inference on the unannotated data you want to classify.
        """

        ####
        # SAHI : Adapt with SAHI script for prediction on new images.
        ####
        if os.path.isfile(os.path.join(self.model_directory, "classes.json")):
            with open(
                os.path.join(self.model_directory, "classes.json"), "r"
            ) as classes:
                labels = json.load(classes)
        else:
            if os.path.isfile(os.path.join(self.train_directory, "train.json")):
                extract_classes_from_coco_json(
                    os.path.join(self.train_directory, "train.json"),
                    os.path.join(self.model_directory, "classes.json"),
                )
            elif os.path.isfile(os.path.join(self.train_directory, "train.json")):
                extract_classes_from_coco_json(
                    os.path.join(self.test_directory, "test.json"),
                    os.path.join(self.model_directory, "classes.json"),
                )
            else:
                print(
                    f"Did not find a classes.json file to map output to class labels in model directory {self.model_directory}."
                )
                print(
                    f"Tried to generate a classes.json file but did not find train.json in training folder: {self.train_directory} or test.json in test folder: {self.test_directory}"
                )
                print("Continue with inference and use numbers instead of class names.")
                labels = None
        # do batch inference
        # returns relative path to resulting JSON
        export_dir = predict(
            model_type="detectron2",
            slice_width=self.slice_width,
            slice_height=self.slice_height,
            overlap_height_ratio=self.overlap_height_ratio,
            overlap_width_ratio=self.overlap_width_ratio,
            source=inference_source_directory,
            model_path=os.path.join(self.model_directory, "model_final.pth"),
            model_config_path=os.path.join(
                self.model_directory, "model_parameters.yaml"
            ),
            no_standard_prediction=True,
            export_visual=False,
            return_dict=True,
            model_category_mapping=labels,
            model_confidence_threshold=self.threshold,
            dataset_json_path=os.path.join(
                inference_source_directory, "inference.json"
            ),
            model_device="cuda",
        )["export_dir"]

        if not os.path.isdir(inference_result_directory):
            os.mkdir(inference_result_directory)
        # Loading the predictions in a DataFrame, deduplicating overlaping predictions
        sahi_result_to_coco(
            os.path.join(
                self.project_directory, os.path.join(export_dir, "result.json")
            ),
            os.path.join(inference_source_directory, "inference.json"),
            os.path.join(
                self.project_directory, os.path.join(export_dir, "result_coco.json")
            ),
        )
        with open(
            os.path.join(
                self.project_directory, os.path.join(export_dir, "result_coco.json")
            ),
            "r",
        ) as j:
            tpred = json.load(j)
        # apply non maximum supression
        df_pred = non_max_supression(
            coco2df(tpred), self.nms_iou_threshold, self.class_agnostic
        )
        # output annotated images
        draw_coco_bbox(
            df_pred, inference_result_directory, inference_source_directory, False
        )
        print("\n---------------Finished Inference---------------")
