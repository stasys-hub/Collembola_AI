import argparse
def get_arguments():
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
        "-i",
        "--input_dir",
        help="""Input directory for inference."""
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        help="""Output directory for inference."""
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

    return parser.parse_args()
