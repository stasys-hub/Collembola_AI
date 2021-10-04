![Screenshot](CAI_git.png)
## Welcome to Collembola_AI
in this project we retrained the popular [Faster R-CNN](https://arxiv.org/pdf/1506.01497.pdf) Convolutional Neural Network using Facebooks [Detectron2](https://github.com/facebookresearch/detectron2) Framework to detect and classify species in a mock community of soil mesofauna preserved in ethanol. The community is made of 10 Collembola species and 2 Acari species, and were acquired using our homemade macrophotography workflow "hoverMacroCam" (described in Sys et al (submitted) contact us for details in the mean time).

1. Sinella curviseta
2. Lepidocyrtus lignorum
3. Desoria tigrina
4. Sphaeridia pumilis
5. Sminthurides aquaticus
6. Megalothorax minimus
7. Folsomia candida
8. Deuterosminthurus bicinctus
9. Ceratophysella gibbosa
10. Xenylla boerneri
11. Malaconothrus monodactylus
12. Hypochthonius rufulus


We provide a wrapper script, collembolAI.py, written in python to perfom inference on ones dataset with only having to change a few lines of code. Furthermore this wrapper script simplifies the registration of custom datasets. Since we provide our own trained model you can retrain this model to extend it's classes and usability. 

Currently we expect the labeled image data to be labeled in COCO format as this immensly simplifies the registration process with Detectron2. For our own labeling workflow we used [labelImg](https://github.com/tzutalin/labelImg) with PascalVOC labels format. The conversion to COCO format can be done with [voc2coco](https://github.com/yukkyo/voc2coco), a version of the script is distributed in this project.

We wrote the code on a machine running Ubuntu 20.04 LTS, but any Linux environment allowing you to run Pytorch with proprietary Nvidia drivers (tested with 450.102.04) and CUDA (tested with 11.0 & 10.2) should work. MacOS and Windows should also work if you get Detectron running. 

### Dependencies

1. Python 3 
2. Detectron 2 - since we use the Detectron2 API we have the same Dependecies. Please refer to their [Documentation](https://detectron2.readthedocs.io/tutorials/install.html#requirements). 
3. CUDA enabled Nvidia GPU

### Installation

Suggestion on how to install Detectron2:

```bash
# create a python env with python version 3.8
python3.8 -m venv path/to/env
# activate env
source activate path/to/env/bin/activate
# install pytorch and torchvision for CUDA 11.0 (may differ -> depends on your setup) 
# pytorch 1.8 seems to be problematic with detectron2 currently, using pytorch 1.7 should solves this.
pip install torch==1.7.0+cu111 torchvision==0.8.1+cu110 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
# install Detectron2 
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html

To use CollembolAI just clone the git Directory into a place of your choice. 

### Usage

The Script collembolAI.py shipped with CollembolAI is the heart of the pipeline. The simplest way to get started on a new dataset is:

1. to create a project directory (e.g. /home/clem/myproject), a train directory (/home/clem/myproject/train) and a test directory (/home/clem/myproject/test)
2. Directly inside this folder, split your annotated pictures (jpg) into a train set folder named 'train' and a test set folder named 'test' (You must have something like /home/clem/myproject/train and /home/clem/myproject/test).
3. Split the COCO labels accordingly (/home/clem/myproject/train/train.json and /home/clem/myproject/test/test.json)
4. Create a new template (simple text file) to set up the retraining of a model and give it a name. You can reuse and adapt the template provided with CollembolAI (template.conf).
5. In the template, make sure to indicate the path to you project folder, and give a name to your model
6. The template default to the faster rcnn R50 FPN 3x model, that should work on most modern laptop equiped with a good NVDIA GPU). To select another base model provided by Detectron2, please check the [model zoo](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md) and change "model_zoo_config" in the configuration file accordingly. The  Faster RCNN models should work, provided your hardware can follow.
7. If you want to run your model on new, unannotated pictures, drop them in a new folder and indicate this folder name in the configuration file using 'inference_directory' (e.g. /home/clem/myproject/new_pictures_to_classify)
8. For the other optional configuration variable, please check Detectron2 documentation

```
[DEFAULT]
project_directory = /home/clem/myproject
model_name = mymodel_using_faster_rcnn_R50_FPN
 
[OPTIONAL]
inference_directory = new_pictures_to_classify
model_zoo_config = COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml
iterations = 8000
number_of_workers = 2
batch_size = 5
learning_rate = 0.00025 
detection_treshold = 0.6
gpu_device_num = 0
```

You are all set to train and test a model and then use it on new pictures:

1. describe the training set
```
python3 collembolAI.py -d template.conf
```
1. training
```
python3 collembolAI.py -t template.conf
```
2. testing
```
python3 collembolAI.py -e template.conf
```
3. annotate new pictures
```
python3 collembolAI.py -a template.conf
```
4. activate all functions:
```
python3 collembolAI.py -d -t -e -a template.conf
```


<!--
```python
# Please declare your working and output directory for training and test set here. 
# this is were your "train" and output folder should be
my_work_dir = "/path/to/working/dir"
# were things will be stored
my_output_dir = "/path/to/output/dir"

```
You can specify Model parameters and use the model using the following syntax:
```python
# Start model with default params  and  a learning rate of 0.00001
 My_Model = collembola_ai(my_work_dir, my_output_dir, learning_rate=0.00001)
 
 # this will print the current model configuration
 My_Model.print_model_values()
 
 # register the training and My_Model.sets in detectron2
 My_Model.load_train_test()
 
 # start training 
 My_Model.start_training()
 
 # start evaluation on My_Model.set
 My_Model.start_evaluation_on_test()
 
 # run inference on a specified dataset
 # specifiy image type 
 my_type = "jpg"
 # path to outout for annotated images 
 my_output_inference = "/path/to/..."
 My_Model.perfom_inference_on_folder(imgpath, my_output_inference, my_type)
```
Currently supported parameters and their defaults are:
  1. workingdir: path to working directory (str) - required
  2. outputdir : path to output directory (str) - required
  3. n_iterations: Number of iterations to train (int = 8000)
  4. work_num: Number of workes (int = 2)
  5. my_batch_size: Batch size used for training (int = 5) Should work with any Nvidia GPU >= 8GB VRAM
  6. learning_rate: The models learning rate (float = 0.00025)
  7. number_classes: number of classes defined in the training data set (int = 10) - adjust to number of labeled species you have!
  8. treshhold: Treshhold for Detection in evaluation (float = 0.55)
  
Please read the prequisites in the script header. They contain information on how to structure the your directory tree. 
Please don't forget that you can either perform inference with an already trained model (shipped with CollembolaAI) or you can choose to retrain it on your data as well as start from scratch and train a new model. If you want to train/retrain your model you will need a data-set labeled in COCO Format.

### TODOs:
Split the script and mnake it modular



Since we don't use segemntation and only want bounding boxes the easiest way to annotate your data is to use [labelimg](https://pypi.org/project/labelImg/) and convert the data to COCO format with a short python script. 
-->
