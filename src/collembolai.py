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
import os
from cai_model import collembola_ai
from utils.parser import get_arguments
from utils.describe_dataset import describe_train_test


def main():

    # get commandline arguments
    args = get_arguments()

    # set some globals for CUDA Device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpu

    # load configuration
    My_Model = collembola_ai(config_path=args.config_file)

    # print the model parameters
    My_Model.print_model_values()

    # register the training and My_Model.sets in detectron2
    My_Model.load_train_test()

    if args.sets_description:
        # TODO add save state flag
        describe_train_test(
            My_Model.project_directory,
            My_Model.train_directory,
            My_Model.test_directory,
        )

    # start training
    if args.train:
        My_Model.start_training()
    else:
        print("-t not set -> skipping training")

    # start evaluation on test set
    if args.evaluate:
        My_Model.start_evaluation_on_test(
            nms_iou_threshold=My_Model.nms_iou_threshold
        )
    else:
        print("-e not set -> skipping evaluation")

    # annotate images in given folder
    if args.annotate:
        if args.input_dir is None:
            raise ValueError("Specify input directory with '-i' or '--input_dir'.")
        else:
            input_dir = args.input_dir
        if args.output_dir is None:
            output_dir = os.path.join(input_dir,"results")
            print(f"No output directory set. Results will be saved in {output_dir}")
        else:
            output_dir = args.output_dir
        # Run inference with your trained model on unlabeled data
        My_Model.perform_inference_on_folder(
            output_dir,
            input_dir
        )
    else:
        print("-a not set -> skipping annotation")


if __name__ == "__main__":
    main()
