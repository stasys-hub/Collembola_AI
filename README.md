![Screenshot](CAI_git.png)
## Welcome to Collembola_AI
in this project we retrained the popular [Faster R-CNN](https://arxiv.org/pdf/1506.01497.pdf) Convolutional Neural Network using Facebooks [Detectron2](https://github.com/facebookresearch/detectron2) Framework to detect 9 different Collembola Species. This model can be used currently to classify the following species:
1. Sinella
2. Pseudosinella
3. Desoria
4. Sphaeridia
5. Sminthurides
6. Megalothorax
7. Folsomia
8. Deuterosminthurus
9. Ceratophysella


We provide a wrapper script written in python to perfom inference on ones dataset with only having to change a few lines of code. Furthermore this wrapper script simplifies the registration of custom datasets. Since we provide our own trained model you can retrain this model to extend it's classes and usability. 

Currently we expect the labeled image data to be labeled in COCO Format as this immensly simplifies the registration process with Detectron2. We plan to provide conversion scripts from other common formats in the near future. 

We wrote the code on a machine running Ubuntu 20.04 LTS, but any Linux environment allowing you to run Pytorch with proprietary Nvidia drivers (tested with 450.102.04) and CUDA (tested with 11.0 & 10.2) should work. MacOS and Windows should also work if you get Detectron running. 

### Dependencies

1. Python 3 
2. Detectron 2 - since we use the Detectron2 API we have the same Dependecies. Please refer to their [Documentation](https://detectron2.readthedocs.io/tutorials/install.html#requirements). 
3. CUDA enabled Nvidia GPU

### Installation

Suggusttion on how to install Detectron2:

```bash
# create a python env with python version 3.8
python3.8 -m venv path/to/env
# activate env
source activate path/to/env/bin/activate
# install pytorch and torchvision for CUDA 11.0 (may differ -> depends on your setup) 
# pytorch 1.8 ssems to make poblems with detectron2 currently, so i would recommend to use 1.7
pip install torch==1.7.0+cu111 torchvision==0.8.1+cu110 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
# install Detectron2 
python -m pip install detectron2 -f   https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html
```

To use CollembolaAI just clone the git Directory into a place of your choice. 

### Usage

The Script collembolaAI_alpha.py shipped with CollembolAI is the heart of the project. It should work right of the box after you have adapted the path' in the main() function of the script. 

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
