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
import argparse
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

    # start training with duster
    if args.train_duster:
        My_Model.start_training_duster(epochs=50)
    else:
        print("-d not set -> skipping duster training")

    
    # start evaluation on test set
    if args.evaluate:
        My_Model.start_evaluation_on_test(
            dedup_thresh=My_Model.dedup_thresh, dusting=My_Model.duster
        )
    else:
        print("-e not set -> skipping evaluation")

    # annotate images in given folder
    if args.annotate:
        # Run inference with your trained model on unlabeled data
        My_Model.perform_inference_on_folder(
            imgtype="jpg", dusting=My_Model.duster, dedup_thresh=My_Model.dedup_thresh
        )
    else:
        print("-a not set -> skipping annotation")


if __name__ == "__main__":
    main()