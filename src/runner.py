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

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "config_file",
        type=str,
        help="""Path of the configuration file (default: "./CAI.conf")""",
    )

    parser.add_argument(
        "-t",
        "--train",
        action="store_true",
        help="""(re-)Train a model using the train set of pictures (default: skip)""",
    )

    parser.add_argument(
        "-d",
        "--train_duster",
        action="store_true",
        help="""(re-)Train the CNN "duster" using the train set of pictures (require a trained rCNN first, default: skip)""",
    )

    parser.add_argument(
        "-e",
        "--evaluate",
        action="store_true",
        help="""Evaluate the model using the test set of pictures (default: skip)""",
    )

    parser.add_argument(
        "-a",
        "--annotate",
        action="store_true",
        help="""Annotate the inference set of pictures (default: skip)""",
    )

    parser.add_argument(
        "-s",
        "--sets_description",
        action="store_true",
        help="""Output some descriptions elements for the train and test set in the project directory""",
    )

    parser.add_argument(
        "--visible_gpu",
        type=str,
        default="0",
        help="""List of visible gpu to CUDA (default: "0", example: "0,1")""",
    )

    parser.add_argument(
        "--gpu_num",
        type=int,
        default=0,
        help="""Set the gpu device number to use (default: 0)""",
    )

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpu

    # Example: Run CollembolAI with your defined parameters
    # Define your model parameters
    My_Model = collembola_ai(config_path=args.config_file)

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
        My_Model.start_evaluation_on_test(
            dedup_thresh=My_Model.dedup_thresh, dusting=My_Model.duster
        )
    else:
        print("Skipping evaluation")

    if args.annotate:
        # Run inference with your trained model on unlabeled data
        My_Model.perform_inference_on_folder(
            imgtype="jpg", dusting=My_Model.duster, dedup_thresh=My_Model.dedup_thresh
        )
    else:
        print("Nothing to annotate")


if __name__ == "__main__":
    main()
