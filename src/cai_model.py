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

#import postprocess.duster as duster


from utils.cocoutils import (
    testresults2coco,
    coco2df,
    create_coco_json_for_inference,
    sahi_result_to_coco
)

from utils.output_inference_images import draw_coco_bbox
from postprocess.nms import non_max_supression
from evaluate.match_groundtruth import match_true_n_pred_box

from PIL import Image
import numpy as np

from sklearn.metrics import confusion_matrix
from utils.third_party_utils import plot_confusion_matrix

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
        self.inference_directory = os.path.join(
            self.project_directory, config["OPTIONAL"]["inference_directory"]
        )
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
        with open(os.path.join(self.train_directory, "train.json"), "r") as js:
            self.num_classes = len(json.load(js)["categories"])
            print(f"Found {self.num_classes} classes in the training annotation file")


        # set gpu device to use
        self.gpu_num = int(config["OPTIONAL"]["gpu_device_num"])



        self.nms_iou_threshold = float(config["OPTIONAL"]["non_maximum_supression_iou_threshold"])

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
            raise FileNotFoundError(f"Did not find train directory in project folder. Please provide train directory with path {self.train_directory}.")
        if not os.path.isfile(os.path.join(self.train_directory,"train.json")):
            raise FileNotFoundError("Please provide a coco annotation file 'train.json'.")
        unsliced_train_directory = self.train_directory + "_unscliced"
        shutil.move(self.train_directory,unsliced_train_directory)
        os.mkdir(self.train_directory)
        slicing.slice_coco(coco_annotation_file_path = os.path.join(unsliced_train_directory,"train.json"),
                image_dir = unsliced_train_directory,
                output_coco_annotation_file_name = os.path.join(self.train_directory,"train.json"),
                output_dir = self.train_directory,
                slice_height = self.slice_height,
                slice_width = self.slice_width,
                overlap_height_ratio = self.overlap_height_ratio,
                overlap_width_ratio = self.overlap_width_ratio)
        

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
        register_coco_instances(
            "test", {}, os.path.join(self.test_directory, "test.json"), "test"
        )

    def start_training(self):
        '''This function will configure Detectron with your input Parameters and start the Training.
        HINT: If you want to check your Parameters before training use "print_model_values"'''

        ##
        ## UPDATE WITH THE TRAINING SCRIPT USING SAHI
        ##
        # check if the train images are sliced
        Image.MAX_IMAGE_PIXELS = None
        with open(os.path.join(self.train_directory,"train.json"),"r") as train_json:
            train_coco = json.load(train_json)
        im = Image.open(os.path.join(self.train_directory,train_coco["images"][0]["file_name"]))
        width, height = im.size
        if width > self.slice_width or height > self.slice_height:
            print("----------------------------------")
            print("----------------------------------")
            print("Loaded one image from train directory")
            print(train_coco['images'][0]['file_name'])
            print(f"width: {width}px \t height: {height}px")
            print("Config file specifies")
            print(f"width: {self.slice_width}px \t height: {self.slice_height}px")
            print("----------------------------------")
            print("----------------------------------")
            raise ValueError("The size of the images in the train directory is larger than specified in the config file. Please run slicing first (flag '--slice') and ensure that the train directory is named 'train'.")
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
        with open(os.path.join(self.model_directory,"model_parameters.yaml"),"w") as outfile:
            configuration = cfg.dump().split("\n")
            # remove unneeded configurations that are not needed for SAHI (and lead to errors)
            for idx,parameter in enumerate(configuration):
                if "DEVICE" in parameter or "nms" in parameter:
                    configuration.pop(idx)
            configuration = "\n".join(configuration)
            outfile.write(configuration)
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
        cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
        trainer.train()
        print("\n---------------Finished Training---------------")


    def start_evaluation_on_test(self, nms_iou_threshold=0.15, verbose=True):
        """This function will run the trained model on the test dataset (test/test.json)"""

        ##
        ## UPDATE WITH THE TEST INFERENCE SCRIPT USING SAHI
        ##

        # RUNNING INFERENCE
        # ================================================================================================

        # TO-DO WRITE A JSON FILE WITH LABELS WHEN TRAINING
        labels = {"0": "Sminthurides_aquaticus__281415","1":"Folsomia_candida__158441",
        "2":"Lepidocyrtus_lignorum__707889","3":"Xenylla_boerneri__1725404",
        "4":"Sphaeridia_pumilis__212016","5":"Megalothorax_minimus__438500",
        "6":"Ceratophysella_gibbosa__187618","7":"Ceratophysella_gibbosa__187618",
        "8":"Malaconothrus_monodactylus__229876","9":"Sinella_curviseta__187695",
        "10":"Deuterosminthurus_bicinctus__2041938","11":"Desoria_tigrina__370036"}
        # do batch inference on test set
        # returns relative path to resulting JSON
        export_dir = "runs/predict/exp5"
        """export_dir = predict(model_type="detectron2", 
                            slice_width=self.slice_width, 
                            slice_height=self.slice_height, 
                            overlap_height_ratio=self.overlap_height_ratio, 
                            overlap_width_ratio=self.overlap_width_ratio, 
                            source= self.test_directory,
                            model_path=os.path.join(self.model_directory,"model_final.pth"),
                            model_config_path=os.path.join(self.model_directory,"model_parameters.yaml"),
                            no_standard_prediction=True,
                            export_visual=False,
                            return_dict=True, 
                            model_category_mapping=labels,
                            model_confidence_threshold=self.threshold,
                            dataset_json_path=os.path.join(self.test_directory, "test.json"),
                            model_device="cuda"
        )["export_dir"]"""

        # RUNNING EVALUATION
        # ================================================================================================
        print("Reporting and evaluating the inference on test set",end="\n\n\n")
        print('Loading predicted labels from "result.json"')
        if not os.path.isdir(os.path.join(self.project_directory, "test_results")):
            os.mkdir(os.path.join(self.project_directory, "test_results"))
        # Loading the predictions in a DataFrame, deduplicating overlaping predictions
        tpred = testresults2coco(self.test_directory, os.path.join(self.project_directory,export_dir), write=True)
        df_pred = non_max_supression(coco2df(tpred), nms_iou_threshold)

        # Loading train set and test set in DataFrame
        with open(os.path.join(self.test_directory, "test.json"), "r") as j:
            ttruth = json.load(j)
            df_ttruth = coco2df(ttruth)
            df_ttruth["id_true"] = df_ttruth["id"]

        with open(os.path.join(self.train_directory, "train.json"), "r") as j:
            train = json.load(j)
            df_train = coco2df(train)
            df_train["id_train"] = df_train["id"]

        # Computing representation (abundance and area) of each classes in the train and test dataset
        # ------------------------------------------------------------------------------------------------
        tt_abundances = (
            df_train.name.value_counts()
            .to_frame()
            .join(df_ttruth.name.value_counts(), lsuffix="_train", rsuffix="_test")
        )
        tt_abundances.columns = ["Train", "Test"]
        tt_abundances = (
            tt_abundances.join(df_pred.name.value_counts())
            .join(df_ttruth.groupby("name").sum()["area"])
            .join(df_pred.groupby("name").sum()["area"], rsuffix="pred")
        )
        tt_abundances.columns = [
            "Train",
            "Test True",
            "Test Pred",
            "Test True Area",
            "Test Pred Area",
        ]
        tt_abundances["Perc Pred True"] = (
            tt_abundances["Test Pred Area"] / tt_abundances["Test True Area"] * 100
        )
        tt_abundances["Test True Contribution To Total Area"] = (
            tt_abundances["Test True Area"]
            / tt_abundances["Test True Area"].sum()
            * 100
        )
        tt_abundances["Test Pred Contribution To Total Area"] = (
            tt_abundances["Test Pred Area"]
            / tt_abundances["Test Pred Area"].sum()
            * 100
        )
        tt_abundances.to_csv(
            os.path.join(
                self.project_directory, os.path.join("test_results","species_abundance_n_area.tsv")
            ),
            sep="\t",
        )
        # ------------------------------------------------------------------------------------------------

        # Matching the predicted annotations with the true annotations
        pairs = match_true_n_pred_box(df_ttruth, df_pred, IoU_threshold=0.4)
        pairs.to_csv(os.path.join(self.project_directory, os.path.join("test_results","pairs.csv")))
        # Computing detection rate, classification accuracy, false positive rate
        # ------------------------------------------------------------------------------------------------
        total_true_labels = pairs.id_true.notnull().sum()
        true_labels_without_matching_preds = pairs.id_pred.isnull().sum()
        perc_detected_animals = 100 - (
            true_labels_without_matching_preds / total_true_labels * 100
        )
        perc_correct_class = (
            pairs["is_correct_class"].sum() / pairs.dropna().shape[0] * 100
        )

        if verbose:
            print(f"The test set represents a total of {total_true_labels} specimens.")
            print(
                f'The model produced {len(tpred["annotations"])} prediction, of which {df_pred.shape[0]} remains after deduplication'
                + " and removal of oversized bounding boxes."
            )
            print(
                f"{total_true_labels - true_labels_without_matching_preds} ({round(perc_detected_animals, 1)}% of the total) "
                + "of the actual specimens were correcly detected."
                + f' Of those detected specimens, {int(pairs["is_correct_class"].sum())} (= {round(perc_correct_class, 1)}%) where assigned to the correct species.'
            )

        # Tagging the false positives in df_pred
        df_pred = df_pred.merge(pairs[["id_pred", "id_true"]], how="left", on="id_pred")
        df_pred["is_false_positive"] = True
        df_pred["is_false_positive"] = df_pred["is_false_positive"].where(
            df_pred["id_true"].isnull(), False
        )

        # Adding inference outcomes on the true labels, df_ttruth
        df_ttruth = df_ttruth.merge(
            pairs[pairs["name_true"].notnull()][
                ["id_true", "score", "name_pred", "is_correct_class"]
            ],
            on="id_true",
        )
        df_ttruth["is_detected"] = (
            df_ttruth["is_correct_class"]
            .where(df_ttruth["is_correct_class"].isnull(), 1)
            .fillna(0)
        )

        if verbose:
            print(
                f'Of the predicted labels, {df_pred["is_false_positive"].sum()} '
                + f'(={round(df_pred["is_false_positive"].sum() / df_pred.shape[0] * 100,1)}%) '
                + "where false positive (background, not related to a real specimen)"
            )
        # ------------------------------------------------------------------------------------------------

        # Drawing the predicted annotations on the pictures
        # ------------------------------------------------------------------------------------------------
        print(
            "\n\nDrawing the predicted annotations of the test pictures to support visual verification"
        )
        print("Do not use for testing or for training ! =)")
        draw_coco_bbox(
            df_pred,
            os.path.join(self.project_directory, "test_results"),
            self.test_directory,
            True,
            prefix="predicted",
            line_width=10,
            fontsize=150,
            fontYshift=-125,
        )
        # ------------------------------------------------------------------------------------------------

        # Plotting the confusion matrices
        # ------------------------------------------------------------------------------------------------
        # 1. CM including only the detected true label
        mcm = confusion_matrix(
            pairs.dropna().name_true,
            pairs.dropna().name_pred.fillna("NaN"),
            labels=pairs.dropna().name_true.unique(),
        )
        plot_confusion_matrix(
            mcm,
            pairs.dropna().name_true.unique(),
            write=os.path.join(
                self.project_directory, "test_results/cm_onlydetected.png"
            ),
        )

        # 2. CM including only the detected true label, normalized
        # Note: the normalized matrix option is bugged in the plot_confusion_matrix function from sklearn
        # Thus I normalize the matrix here before plotting and don't use the option
        mcm = mcm.astype("float") / mcm.sum(axis=1)[:, np.newaxis] * 100
        mcm = mcm.round(1)
        plot_confusion_matrix(
            mcm,
            pairs.dropna().name_true.unique(),
            write=os.path.join(
                self.project_directory, "test_results/cm_norm_onlydetected.png"
            ),
        )

        # 3. CM including only the undetected true label (Nan)
        mcm = confusion_matrix(
            pairs.name_true.fillna("NaN"),
            pairs.name_pred.fillna("NaN"),
            labels=pairs.fillna("NaN").name_true.unique(),
        )
        plot_confusion_matrix(
            mcm,
            np.append(pairs.name_true.unique(), "NaN"),
            write=os.path.join(self.project_directory, "test_results/cm_inclNaN.png"),
        )


        # 4. CM including only the undetected true label (Nan), normalized
        
        mcm = mcm.astype("float") / mcm.sum(axis=1)[:, np.newaxis] * 100
        mcm = np.nan_to_num(mcm.round(1))
        plot_confusion_matrix(
            mcm,
            np.append(pairs.name_true.unique(), "NaN"),
            write=os.path.join(
                self.project_directory, "test_results/cm_norm_inclNaN.png"
            ),
        )
        print("\n---------------Finished Evaluation---------------")
        

        # ================================================================================================

    def perform_inference_on_folder(
        self,
        inference_source_directory: str,
        inference_result_directory:str):
        '''
        This function can be used to perform inference on the unannotated data you want to classify.
        '''

        if not inference_source_directory:
            inference_source_directory = self.inference_directory

        ####
        # SAHI : Adapt with SAHI script for prediction on new images.
        ####
        # TO-DO WRITE A JSON FILE WITH LABELS WHEN TRAINING
        labels = {"0": "Sminthurides_aquaticus__281415","1":"Folsomia_candida__158441",
                  "2":"Lepidocyrtus_lignorum__707889","3":"Xenylla_boerneri__1725404",
                  "4":"Sphaeridia_pumilis__212016","5":"Megalothorax_minimus__438500",
                  "6":"Ceratophysella_gibbosa__187618","7":"Hypochtonius_rufulus__66580",
                  "8":"Malaconothrus_monodactylus__229876","9":"Sinella_curviseta__187695",
                  "10":"Deuterosminthurus_bicinctus__2041938","11":"Desoria_tigrina__370036"}
        # write a coco-style json file, to enable export of labels
        create_coco_json_for_inference(inference_source_directory, labels)

        # do batch inference on test set
        # returns relative path to resulting JSON
        export_dir = predict(model_type="detectron2", 
                            slice_width=self.slice_width, 
                            slice_height=self.slice_height, 
                            overlap_height_ratio=self.overlap_height_ratio, 
                            overlap_width_ratio=self.overlap_width_ratio, 
                            source=inference_source_directory,
                            model_path=os.path.join(self.model_directory,"model_final.pth"),
                            model_config_path=os.path.join(self.model_directory,"model_parameters.yaml"),
                            no_standard_prediction=True,
                            export_visual=False,
                            return_dict=True, 
                            model_category_mapping=labels,
                            model_confidence_threshold=self.threshold,
                            dataset_json_path=os.path.join(inference_source_directory, "inference.json"),
                            model_device="cuda"
        )["export_dir"]

        if not os.path.isdir(inference_result_directory):
            os.mkdir(inference_result_directory)
        # Loading the predictions in a DataFrame, deduplicating overlaping predictions
        sahi_result_to_coco(os.path.join(self.project_directory,os.path.join(export_dir,"result.json")),
                            os.path.join(inference_source_directory, "inference.json"),
                            os.path.join(self.project_directory,os.path.join(export_dir,"result_coco.json")))
        with open(os.path.join(self.project_directory,os.path.join(export_dir,"result_coco.json")),"r") as j:
            tpred = json.load(j)
        # apply non maximum supression
        df_pred = non_max_supression(coco2df(tpred), self.nms_iou_threshold)
        # output annotated images
        draw_coco_bbox(df_pred, inference_result_directory, inference_source_directory, False)
        print("\n---------------Finished Inference---------------")
