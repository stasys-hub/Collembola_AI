
#required libraries for detectron2
#PyTorch ≥ 1.5 and torchvision that matches the PyTorch installation. You can install them together at pytorch.org to make sure of this
#!pip install torchvision
#OpenCV is optional and needed by demo and visualization
#!pip install opencv-python
#This will only work on Linux or macOS
####if Detectron2 is missing:
#!python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'


import os
import numpy as np
import json
import random
import cv2
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.structures import BoxMode

#set here your variables

#train: Annotated Images for training
#train.json: Annotations + Path to Images for all images stored in train
#test: Annotated Images for model evaluation (Taken from train)
#test.json: Like train.json but for test
#test_set2: Unannotated images which do not stem from the train data to evaluate generalization of the model

#please ensure the following structure:
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

#working directory 
def train_my_model(workingdir, outputdir, n_iterations=5000, work_num=2, my_batch_size=4, learning_rate=0.00025, number_classes=10, treshhold=0.55 ):
    working_dir=workingdir
    #Save Model Weights to
    output_dir_model=outputdir
    #save pictures with prediction
    output_dir_imgs=outputdir
    ############################################
    ####OPTIMIZE HERE FOR BETTER PERFORMANCE####
    #Iterations for training
    training_iterations=n_iterations
    #Number of workers
    worker_num=work_num
    #Images per Batch (a higher number will need more VRAM, but is also good for stability in the learning process)
    batchsize=my_batch_size
    #Base learning rate
    base_lr=learning_rate
    #Number of classes (always +1 bc of background)
    num_classes=number_classes
    #detection threshold (threshold for a class probability upon which a prediction will be considered to be true)
    detect_thresh=treshhold
    #if test_set2 was renamed
    test_set2="test_set2"


    #####DO NOT CHANGE#####
    train_dir=os.path.join(working_dir,"train")
    train_json="train.json"

    json_train_file = os.path.join(working_dir, "train", train_json)
    #load dataset and create coco dataset for DETECTRON2
    with open(json_train_file) as f:
        imgs_anns = json.load(f)

    register_coco_instances("train", {}, os.path.join(working_dir,"train","train.json"), "train")
    register_coco_instances("test", {}, os.path.join(working_dir,"test","test.json"), "test")

    dataset_dicts = DatasetCatalog.get("train")
    dataset_metadata = MetadataCatalog.get("train")

    #a warning about the category ids will appear - this is okay

    #create the config file for training
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = worker_num
    cfg.OUTPUT_DIR = output_dir_model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = batchsize
    cfg.SOLVER.BASE_LR = base_lr # pick a good LR
    cfg.SOLVER.MAX_ITER = training_iterations
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.nms = True

    #THIS WILL START THE TRAINER, which can be time consuming
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()

    #config for test mode
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = detect_thresh # set the testing threshold for this model
    cfg.DATASETS.TEST = ("test", )
    predictor = DefaultPredictor(cfg)

    #prepare test dataset
    dataset_dicts_test = DatasetCatalog.get("test")
    dataset_metadata_test = MetadataCatalog.get("test")

    #annotate all pictures of testset1 and testset2 with the predictions of the trained model
    os.makedirs(output_dir_imgs, exist_ok=True)
    output_testset1 = os.path.join(output_dir_imgs, "testset1")
    os.makedirs(output_testset1, exist_ok=True)
    output_testset2 = os.path.join(output_dir_imgs, "testset2")
    os.makedirs(output_testset2, exist_ok=True)

    i = 0
    for d in dataset_dicts_test:
        #create variable with output name
        output_name = output_testset1 + "/annotated_" + str(i) + ".jpg"
        #load image
        img = cv2.imread(d["file_name"])
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
        #break

    for i in os.listdir(os.path.join(working_dir,test_set2)):
        if ".jpg" in i:
            file_path = os.path.join(working_dir, test_set2, i)
            im = cv2.imread(file_path)
            outputs = predictor(im)
            v = Visualizer(im[:, :, ::-1], metadata=dataset_metadata, scale=1.)
            instances = outputs["instances"].to("cpu")
            v = v.draw_instance_predictions(instances)
            result = v.get_image()[:, :, ::-1]
            output_name = output_testset2 + "/annotated_" + str(i) + ".jpg"
            write_res = cv2.imwrite(output_name, result)
            #break


    #Input /path/to/input
    input_single_img="/media/leto/Samsung_T3/collembola_ai/mix_plate4.jpg"
    #Output (just name, will be saved in $output_dir_imgs/single_images)
    output_single_name="test.jpg"

    #single image prediction
    #define outputname / path
    single_img_output=os.path.join(output_dir_imgs,"single_images")
    os.makedirs(single_img_output, exist_ok=True)

    output_name = os.path.join(single_img_output,output_single_name)
    #load an image
    img = cv2.imread(input_single_img)
    outputs = predictor(img)
    visualizer = Visualizer(img[:, :, ::-1], metadata=dataset_metadata_test, scale=1.)
    instances = outputs["instances"].to("cpu")
    vis = visualizer.draw_instance_predictions(instances)
    result = vis.get_image()[:, :, ::-1]
    write_res = cv2.imwrite(output_name, result)

    #IN-BUILD MODEL EVALUATION
    #I did not found this really useful since the evaluation takes the accurate position strongly into account, something what we do not care as much about...
    evaluator = COCOEvaluator("test", cfg, False, output_dir=output_dir_imgs)
    val_loader = build_detection_test_loader(cfg, "test")
    inference_on_dataset(trainer.model, val_loader, evaluator)

if __name__ == "__main__":
    train_my_model("/home/leto/Training_C_AI_DATA/normal", "/home/leto/Training_C_AI_DATA/normal/1000", 4000)

