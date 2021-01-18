![Screenshot](CAI_git.png)
## Welcome to Collembola_AI
in this project we retrained the popular [Faster R-CNN](https://arxiv.org/pdf/1506.01497.pdf) Convolutional Neural Network using Facebooks [Detectron2](https://github.com/facebookresearch/detectron2) Framework to detect 10 different Collembola Species. This model can be used currently to classify the following species:
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
2. Detectron 2 - since we use the Detectron2 API we have the same Dependecies. Please refer to thei [Documentation](https://detectron2.readthedocs.io/tutorials/install.html#requirements). 
3. CUDA enabled Nvidia GPU

### Installation

To use CollembolaAI just clone the git Directory into a place of your choice. 

### Usage

The Script collembolaAI_alpha.py shipped with CollembolAI is the heart of the project. It should work right of the box after you have adapted the path' in the main() function of the script. 

```python
s = "Python syntax highlighting"
print s
```

Please read the prequisites in the script header. They contain information on how to structure the your directory tree. 
Please don't forget that you can either perform inference with an already trained model (shipped with CollembolaAI) or you can choose to retrain it on your data as well as start from scratch and train a new model. If you want to train/retrain your model you will need a data-set labeled in COCO Format

Since we don't use segemntation and only want bounding boxes the easies way to annotate your data is to use [labelimg](https://pypi.org/project/labelImg/) and convert the data to COCO format with a short python script. 
