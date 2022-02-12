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

import os
import json
import configparser
import pandas as pd
import cv2
import warnings
import ntpath

import postprocess.duster as duster


from utils.cocoutils import (
    testresults2coco,
    coco2df,
    draw_coco_bbox,
    deduplicate_overlapping_preds,
    match_true_n_pred_box,
    d2_instance2dict,
    df2coco,
)

import PIL
import shutil

from sklearn.metrics import confusion_matrix
from utils.third_party_utils import plot_confusion_matrix

PIL.Image.MAX_IMAGE_PIXELS = 500000000
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset


class collembola_ai:

    def __init__(self, config_path: str):
        """Function to initialize the CollembolaAI main class. These Parameters will be used to configure Detectron2"""

        config = configparser.ConfigParser()
        config.read(config_path)

        # set project directories
        self.project_directory = config["DEFAULT"]["project_directory"]
        self.model_name = config["DEFAULT"]["model_name"]
        self.output_directory = os.path.join(self.project_directory, self.model_name)
        self.train_directory = os.path.join(self.project_directory, "train")
        self.test_directory = os.path.join(self.project_directory, "test")
        self.dust_directory = os.path.join(self.project_directory, "dust")
        self.duster_path = os.path.join(self.project_directory, "duster")
        self.inference_directory = os.path.join(
            self.project_directory, config["OPTIONAL"]["inference_directory"]
        )

        # set model parameters
        self.num_iter = int(config["OPTIONAL"]["iterations"])
        self.num_workers = int(config["OPTIONAL"]["number_of_workers"])
        self.batch_size = int(config["OPTIONAL"]["batch_size"])
        self.learning_rate = float(config["OPTIONAL"]["learning_rate"])

        with open(os.path.join(self.train_directory, "train.json"), "r") as js:
            self.num_classes = len(json.load(js)["categories"])

        print(
            f"Found {self.num_classes} classes in the training annotation file"
        )
        self.threshold = float(config["OPTIONAL"]["detection_threshold"])
        self.model_zoo_config = config["OPTIONAL"]["model_zoo_config"]

        # set gpu device to use
        self.gpu_num = int(config["OPTIONAL"]["gpu_device_num"])
        self.trainer = None

        # Using duster ?
        if config["OPTIONAL"]["duster"] == "True":
            self.duster = True
        else:
            self.duster = False
        self.dedup_thresh = float(config["OPTIONAL"]["deduplication_iou_threshold"])

    def print_model_values(self):
        """This function will print all model parameters which can be set by the user. It is useful if you have path problems.
        Hint: On Windows you will probably have to adjust your path because of backslashes"""

        ###
        # SAHI: may require some update ?
        ###

        print("# --------------- Model Parameters ---------------- #\n")
        print(f"Variable           \tValue\n")
        print(f"Project Dir:        \t{self.project_directory}")
        print(f"Output Dir:         \t{self.output_directory}")
        print(f"Model Zoo:          \t{self.model_zoo_config}")
        print(f"Number iterations:  \t{self.num_iter}")
        print(f"Number of workers:  \t{self.num_workers}")
        print(f"Batch Size:         \t{self.batch_size}")
        print(f"Learning Rate:      \t{self.learning_rate}")
        print(f"Number of classes:  \t{self.num_classes}")
        print(f"GPU device number:  \t{self.gpu_num}")
        print(f"Treshhold:          \t{self.threshold}")
        print("\n# ------------------------------------------------- #")

    def load_train_test(self):
        """This function loads the train.json file and registers your training data using the \"register_coco_instances\" function of Detectron2
        IMPORTANT: Currently it is necessary to use this function before performing inference with a trained model"""

        ###
        # SAHI: may require some update ?
        ###

        try:
            # read train.json file

            with open(os.path.join(self.train_directory, "train.json")) as f:
                imgs_anns = json.load(f)

            # register custom datasets in COCO format
            #                       "my_dataset", "metadata", "json_annotation.json", "path/to/image/dir
            register_coco_instances(
                "train", {}, os.path.join(self.train_directory, "train.json"), "train"
            )
            register_coco_instances(
                "test", {}, os.path.join(self.test_directory, "test.json"), "test"
            )
            dataset_dicts = DatasetCatalog.get("train")
            dataset_metadata = MetadataCatalog.get("train")

        except:
            print(
                'ERROR!\nUnable to load model configurations!\nPlease check your input and use "print_model_values" for debugging '
            )

    def start_training(self):
        '''This function will configure Detectron with your input Parameters and start the Training.
        HINT: If you want to check your Parameters before training use "print_model_values"'''

        ##
        ## UPDATE WITH THE TRAINING SCRIPT USING SAHI
        ##

        # load a model from the modelzoo and initialize model weights and set our model params
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(self.model_zoo_config))
        cfg.DATASETS.TRAIN = ("train",)
        cfg.DATASETS.TEST = ()
        cfg.DATALOADER.NUM_WORKERS = self.num_workers
        cfg.OUTPUT_DIR = self.output_directory
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
        """This function will train the duster (CNN binary classifier to recognize dust or other non animal object)"""

        with open(os.path.join(self.train_directory, "train.json"), "r") as j:
            train = json.load(j)
            df_train = coco2df(train)
            # df_train['id_train'] = df_train['id']

        ######
        # Finding some dust using the trained rCNN model
        self.perform_inference_on_folder(
            inference_directory=self.dust_directory,
            imgtype="jpg",
            dedup_thresh=0.999,
            dusting=False,
        )

        with open(
            os.path.join(self.dust_directory, f"{self.model_name}/inferences.json"), "r"
        ) as j:
            tdust = json.load(j)

        df_dust = coco2df(tdust)
        df_dust["name"] = "Dust"

        # Grabing some pieces of background in the train set (optional, currently no longer in use)
        # extract_random_background_subpictures(df_train, self.train_directory, f'{self.duster_path}/train/Dust', num_subpict_per_pict=200)

        print("Preparing the duster training and validation data")

        duster.dump_training_set(
            self.train_directory,
            self.dust_directory,
            self.duster_path,
            df_train,
            df_dust,
        )
        print("Training and validating the duster")
        duster.train_duster(self.duster_path, self.train_directory, epochs=50)

        print("duster trained")

    def start_dusting(self, df_pred, img_dir):
        """Dusting (identifying and removing False Positive ('dust'))"""

        # Extracting subpictures from the predictions
        print("Extracting and organizing the subpictures for dusting")

        def wipe_dir(path):
            if os.path.exists(path) and os.path.isdir(path):
                shutil.rmtree(path)

        wipe_dir(os.path.join(self.duster_path, "to_predict"))

        os.makedirs(os.path.join(self.duster_path, "to_predict/All"), exist_ok=True)

        for file in df_pred.file_name.unique():
            im = PIL.Image.open(img_dir + "/" + file)

            for name in df_pred["name"].unique():
                os.makedirs(
                    os.path.join(self.duster_path, f"to_predict/{name}"), exist_ok=True
                )

                for raw in df_pred[
                    (df_pred["file_name"] == file) & (df_pred["name"] == name)
                ][["box", "id", "name"]].values:
                    im.crop(raw[0].bounds).save(
                        f"{self.duster_path}/to_predict/All/{raw[1]}.jpg", "JPEG"
                    )
                    im.crop(raw[0].bounds).save(
                        f"{self.duster_path}/to_predict/{raw[2]}/{raw[1]}.jpg", "JPEG"
                    )
            im.close()

        list_classes = list(df_pred.name.unique())
        duster.load_duster_and_classify(self.duster_path, list_classes)
        dust = pd.read_csv(f"{self.duster_path}/dust.csv")
        df_pred = df_pred.merge(dust, on="id")
        df_pred = df_pred[df_pred["dust"] != "Dust"]
        return df_pred

    def start_evaluation_on_test(self, dedup_thresh=0.15, verbose=True, dusting=False):
        """This function will run the trained model on the test dataset (test/test.json)"""

        ##
        ## UPDATE WITH THE TEST INFERENCE SCRIPT USING SAHI
        ##

        # RUNNING INFERENCE
        # ================================================================================================
        # Creating the output directories tree
        os.makedirs(self.output_directory, exist_ok=True)
        output_test_res = os.path.join(self.output_directory, "test_results")
        os.makedirs(output_test_res, exist_ok=True)

        # Loading the model
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(self.model_zoo_config))
        cfg.DATASETS.TRAIN = ("train",)
        cfg.DATALOADER.NUM_WORKERS = self.num_workers
        cfg.OUTPUT_DIR = self.output_directory
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
        cfg.DATASETS.TEST = ("test",)

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
        # ================================================================================================

        ####
        # SAHI: At this stage, the json output of SAHI should be in the output_test_res directory, with the
        # name coco_instances_results.json (if under a different name, then other functions must be adapted)
        ####

        ####
        # SAHI :If above condition is met, the rest of the test pipeline does not need to be modified
        ####

        # REPORTING
        # ================================================================================================
        print("Reporting and evaluating the inference on test set")
        print('\n\nLoading predicted labels from "coco_instances_results.json"')

        # Loading the predictions in a DataFrame, deduplicating overlaping predictions
        tpred = testresults2coco(self.test_directory, self.output_directory, write=True)
        df_pred = deduplicate_overlapping_preds(coco2df(tpred), dedup_thresh)

        # Dusting (identifying and removing False Positive ('dust'))
        if dusting:
            df_pred = self.start_dusting(df_pred, self.test_directory)

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
                self.output_directory, "test_results/species_abundance_n_area.tsv"
            ),
            sep="\t",
        )
        # ------------------------------------------------------------------------------------------------

        # Matching the predicted annotations with the true annotations
        pairs = match_true_n_pred_box(df_ttruth, df_pred, IoU_threshold=0.4)

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
            os.path.join(self.output_directory, "test_results"),
            self.test_directory,
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
            normalize=False,
            write=os.path.join(
                self.output_directory, "test_results/cm_onlydetected.png"
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
            normalize=False,
            write=os.path.join(
                self.output_directory, "test_results/cm_norm_onlydetected.png"
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
            normalize=False,
            write=os.path.join(self.output_directory, "test_results/cm_inclNaN.png"),
        )

        # 4. CM including only the undetected true label (Nan), normalized
        mcm = mcm.astype("float") / mcm.sum(axis=1)[:, np.newaxis] * 100
        mcm = np.nan_to_num(mcm.round(1))
        plot_confusion_matrix(
            mcm,
            pairs.fillna("NaN").name_true.unique(),
            normalize=False,
            write=os.path.join(
                self.output_directory, "test_results/cm_norm_inclNaN.png"
            ),
        )

        print("\n---------------Finished Evaluation---------------")

        # ================================================================================================

    def perform_inference_on_folder(
        self,
        inference_directory=None,
        imgtype="jpg",
        dusting=False,
        dedup_thresh=0.15,
    ):
        '''This function can be used to perform inference on the unannotated data you want to classify.
        IMPORTANT: You still have to load a model using "load_train_test"'''

        if not inference_directory:
            inference_directory = self.inference_directory

        ####
        # SAHI : Adapt with SAHI script for prediction on new images.
        ####

        try:
            # reload the model
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file(self.model_zoo_config))
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
            cfg.nms = True

            # config for test mode
            cfg.MODEL.WEIGHTS = os.path.join(self.output_directory, "model_final.pth")
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold
            cfg.DATASETS.TEST = ("test",)
            cfg.MODEL.DEVICE = self.gpu_num
            # Improving detection on crowed picture ?
            cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 10000
            cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 10000
            cfg.TEST.DETECTIONS_PER_IMAGE = 10000

            predictor = DefaultPredictor(cfg)

            # prepare test dataset for metadata
            dataset_dicts_test = DatasetCatalog.get("test")
            dataset_metadata_test = MetadataCatalog.get("test")
        except Exception as e:
            print(e)
            print(
                "Something went wrong while loading the model configuration. Please Check your path'"
            )
            raise

        ####
        # SAHI : In the block below is the loop that used to execute prediction on each new images (looping on images)
        # then collecting output and concatenating it in a single json file following COCO format
        # Probably need to be adapted with SAHI scripts.
        ####

        inference_out = os.path.join(inference_directory, self.model_name)
        n_files = len(
            ([f for f in os.listdir(inference_directory) if f.endswith(imgtype)])
        )
        counter = 1

        print(
            f"Starting inference ...\nNumber of Files:\t{n_files}\nImage extension:\t{imgtype}"
        )
        print("\n# ------------------------------------------------- #\n")

        # try:
        #     # I added this because Detectron2 uses an deprecated overload -> throws warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            os.makedirs(inference_out, exist_ok=True)
            results_coco = {"images": [], "annotations": []}

            # Loop for predicting on each images starts
            for i in os.listdir(inference_directory):
                if i.endswith(imgtype):
                    file_path = os.path.join(inference_directory, i)
                    print(f"processing [{counter}/{n_files}]", end="\r")
                    im = cv2.imread(file_path)
                    outputs = predictor(im)
                    v = Visualizer(
                        im[:, :, ::-1], metadata=dataset_metadata_test, scale=1.0
                    )
                    instances = outputs["instances"].to("cpu")
                    coco = d2_instance2dict(instances, counter, ntpath.basename(i))
                    # v = v.draw_instance_predictions(instances)
                    # result = v.get_image()[:, :, ::-1]
                    # output_name = f"{inference_out}/annotated_{i}"
                    # write_res = cv2.imwrite(output_name, result)
                    counter += 1

                    results_coco["images"] = results_coco["images"] + coco["images"]
                    results_coco["annotations"] = (
                        results_coco["annotations"] + coco["annotations"]
                    )

            # Wrapping results in COCO JSON.
            annotation_id = 0
            for a in results_coco["annotations"]:
                a["id"] = annotation_id
                annotation_id += 1
            results_coco["type"] = "instances"
            results_coco["licenses"] = ""
            results_coco["info"] = ""

            with open(os.path.join(self.train_directory, "train.json"), "r") as j:
                train = json.load(j)
            results_coco["categories"] = train["categories"]

            df_pred = deduplicate_overlapping_preds(
                coco2df(results_coco), dedup_thresh
            )

            if dusting:
                df_pred = self.start_dusting(results_coco, inference_directory)

            results_coco = df2coco(df_pred)

            with open(os.path.join(inference_out, "inferences.json"), "w") as j:
                json.dump(results_coco, j, indent=4)

        # except Exception as e:
        #     print(e)
        #     print(
        #         'Something went wrong while performing inference on your data. Please check your path directory structure\nUse "print_model_values" for debugging'
        #     )
