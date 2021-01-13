# ------------------------------------------------------------------------------ #
#                                                                                #
# Title:                                                            CollembolaAI #
# Authors:                                      Stephan Weißbach & Stanislav Sys #                                                                              
# Purpose:                                    ML Algorithm to detect Collembolas #                                                                              
# Usage:                                                              See ReadMe #
# Dependencies:                                                       See ReadMe # 
# Last Update:                                                        11.01.2021 #
#                                                                                #
# ------------------------------------------------------------------------------ #


# Imports
import os
import numpy as np
import json
import random
import cv2
import time
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.structures import BoxMode


# please ensure the following structure:
# |
# |-working directory
#     |-train
#     |   |-img1.jpg
#     |   |-img2.jpg
#     |   |-[...]
#     |   |-train.json
#     |
#     |-test
#     |   |-img1.jpg
#     |   |-img2.jpg
#     |   |-[...]
#     |   |-test.json
#     |
#     |-test_set2
#         |-img1.jpg
#         |-img2.jpg
#         |-[...]


# train: Annotated Images for training
# train.json: Annotations + Path to Images for all images stored in train
# test: Annotated Images for model evaluation (Taken from train)
# test.json: Like train.json but for test
# test_set2: Unannotated images which do not stem from the train data to evaluate generalization of the model

# OOP Version of CollembolaAI

class collembola_ai:


    def __init__(self, workingdir: str, outputdir: str, n_iterations: int = 8000, work_num: int = 2, my_batch_size: int = 5, learning_rate: float = 0.00025, number_classes:int = 10, treshhold: float = 0.55):
        """Function to initialize the CollembolaAI main class. These Parameters will be used to configure Detectron2"""

        # set model parameters
        self.working_directory = workingdir
        self.output_directory = outputdir
        self.num_iter = n_iterations
        self.num_workers = work_num
        self.batch_size = my_batch_size
        self.learning_rate = learning_rate
        self.num_classes = number_classes
        self.threshold = treshhold
        self.train_directory = os.path.join(workingdir, "train")
        self.test_directory = os.path.join(workingdir, "test")



    def print_model_values(self):
        """This function will print all model parameters which can be set by the user. It is useful if you have path problems.
        Hint: On Windows you will probably have to adjust your path because of backslashes"""

        print("# --------------- Model Parameters ---------------- #\n")
        print(f'Variable           \tValue\n')
        print(f'Working Dir:        \t{self.working_directory}')
        print(f'Output Dir:         \t{self.output_directory}')
        print(f'Number iterations:  \t{self.num_iter}')
        print(f'Number of workers:  \t{self.num_workers}')
        print(f'Batch Size:         \t{self.batch_size}')
        print(f'Learning Rate:      \t{self.learning_rate}')
        print(f'Number of classes:  \t{self.num_classes}')
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


    def start_training(self):
        """This function will configure Detectron with your input Parameters and start the Training.
        HINT: If you want to check your Parameters before training use \"print_model_values\""""

        # load a model from the modelzoo and initialize model weights and set our model params
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.DATASETS.TRAIN = ("train",)
        cfg.DATASETS.TEST = ()
        cfg.DATALOADER.NUM_WORKERS = self.num_workers
        cfg.OUTPUT_DIR =  self.output_directory
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        cfg.SOLVER.IMS_PER_BATCH = self.batch_size
        cfg.SOLVER.BASE_LR = self.learning_rate
        cfg.SOLVER.MAX_ITER = self.num_iter
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
        cfg.nms = True

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
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.DATASETS.TRAIN = ("train",)
        cfg.DATALOADER.NUM_WORKERS = self.num_workers
        cfg.OUTPUT_DIR =  self.output_directory
        cfg.SOLVER.IMS_PER_BATCH = self.batch_size
        cfg.SOLVER.BASE_LR = self.learning_rate
        cfg.SOLVER.MAX_ITER = self.num_iter
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
        cfg.nms = True

        #config for test mode
        cfg.MODEL.WEIGHTS = os.path.join(self.output_directory, "model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold
        cfg.DATASETS.TEST = ("test", )
        predictor = DefaultPredictor(cfg)

        #prepare test dataset
        dataset_dicts_test = DatasetCatalog.get("test")
        dataset_metadata_test = MetadataCatalog.get("test")

        #annotate all pictures of testset1 and testset2 with the predictions of the trained model
        os.makedirs(self.output_directory, exist_ok=True)
        output_testset1 = os.path.join(self.output_directory, "testset1")
        os.makedirs(output_testset1, exist_ok=True)
        output_testset2 = os.path.join(self.output_directory, "testset2")
        os.makedirs(output_testset2, exist_ok=True)


        try:
            i = 0
            for d in dataset_dicts_test:
                #create variable with output name
                output_name = output_testset1 + "/annotated_" + str(i) + ".jpg"
                img = cv2.imread(d["file_name"])
                print(f"Processing: \t{output_name}")
                #make prediction
                outputs = predictor(img)
                #draw prediction
                visualizer = Visualizer(img[:, :, ::-1], metadata=dataset_metadata_test, scale=1.)
                instances = outputs["instances"].to("cpu")
                vis = visualizer.draw_instance_predictions(instances)
                result = vis.get_image()[:, :, ::-1]
                #write image
                write_res = cv2.imwrite(output_name, result)
                i += 1
        except:
            print("Something went wrong while evaluating the \"test\" set. Please check your path directory structure\nUse \"print_model_values\" for debugging")


        try:
            for i in os.listdir(os.path.join(self.working_directory,"JPG")):
                if i.endswith("jpg"):
                    file_path = os.path.join(self.working_directory, "JPG", i)
                    print(f"Processing: \t{file_path}")
                    im = cv2.imread(file_path)
                    outputs = predictor(im)
                    v = Visualizer(im[:, :, ::-1], metadata=dataset_metadata_test, scale=1.)
                    instances = outputs["instances"].to("cpu")
                    v = v.draw_instance_predictions(instances)
                    result = v.get_image()[:, :, ::-1]
                    output_name = output_testset2 + "/annotated_" + str(i) + ".jpg"
                    write_res = cv2.imwrite(output_name, result)
        except:
            print("Something went wrong while performing inference on your data. Please check your path directory structure\nUse \"print_model_values\" for debugging")

        print("\n---------------Finished Evaluation---------------")

    def perfom_inference_on_folder(self, path_to_images,path_to_outputdir: str, imgtype: str):
        """This function can be used to test a trained model with a set of images or to perform inference on data you want to classify.
        IMPORTANT: You still have to load a model using \"load_train_test\""""
        try:
            # reload the model
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
            cfg.DATASETS.TRAIN = ("train",)
            cfg.DATALOADER.NUM_WORKERS = self.num_workers
            cfg.OUTPUT_DIR =  self.output_directory
            cfg.SOLVER.IMS_PER_BATCH = self.batch_size
            cfg.SOLVER.BASE_LR = self.learning_rate
            cfg.SOLVER.MAX_ITER = self.num_iter
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
            cfg.nms = True

            #config for test mode
            cfg.MODEL.WEIGHTS = os.path.join(self.output_directory, "model_final.pth")
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold
            cfg.DATASETS.TEST = ("test", )
            predictor = DefaultPredictor(cfg)

            #prepare test dataset for metadata
            dataset_dicts_test = DatasetCatalog.get("test")
            dataset_metadata_test = MetadataCatalog.get("test")
        except:
            print("Something went wrong while loading the model configuration. Please Check your path\'")


        n_files = len(([f for f in os.listdir(path_to_images) if f.endswith(imgtype)]))
        counter = 1
        print(f"Starting inference ...\nNumber of Files:\t{n_files}\nImage extension:\t{imgtype}")

        try:
            os.makedirs(path_to_outputdir, exist_ok=True)
            for i in os.listdir(path_to_images):
                if i.endswith(imgtype):
                    file_path = os.path.join(path_to_images, i)
                    print(f"processing [{counter}/{n_files}]")
                    im = cv2.imread(file_path)
                    outputs = predictor(im)
                    v = Visualizer(im[:, :, ::-1], metadata=dataset_metadata_test, scale=1.)
                    instances = outputs["instances"].to("cpu")
                    v = v.draw_instance_predictions(instances)
                    result = v.get_image()[:, :, ::-1]
                    output_name = f"{path_to_outputdir}/annotated_{i}"
                    write_res = cv2.imwrite(output_name, result)
                    counter += 1
        except:
            print("Something went wrong while performing inference on your data. Please check your path directory structure\nUse \"print_model_values\" for debugging")


if __name__ == "__main__":

    # stuff do declare 
    my_work_dir = "/home/vim_diesel/Collembola_AI/Training_C_AI_DATA/svd/"
    my_output_dir = "/home/vim_diesel/Collembola_AI/Training_C_AI_DATA/svd/8k_batch10_svd/"


    # run a test
    test = collembola_ai(my_work_dir, my_output_dir, work_num=5)
    test.print_model_values()
    test.load_train_test()
    #test.start_training()
    #test.start_evaluation_on_test()
    # Inference test

    imgpath = "/home/vim_diesel/Collembola_AI/JPG"
    my_output_inference = "/home/vim_diesel/Collembola_AI/testoutput"
    my_type = "jpg"
    #test.perfom_inference_on_folder(imgpath, my_output_inference, my_type)
