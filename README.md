![Screenshot](CAI_git.png)
## Welcome to Collembola_AI
in this project we retrained the popular [Faster R-CNN](https://arxiv.org/pdf/1506.01497.pdf) Convolutional Neural Network using Facebooks [Detectron2](https://github.com/facebookresearch/detectron2) Framework to detect 10 different Collembola Species. This model can be used currently to classify the following species:
1. Sinella
2. Pseudo Sinella
3. Desoria
4. ...

We provide a wrapper script written in python to perfom inference on ones dataset with only having to change a few lines of code. Furthermore this wrapper script simplifies the registration of custom datasets. Since we provide our own model you can retrain this model to extend it's classes and usability. 

Currently we expect the labeled image data to to labeled in COCO Format as this immensly simplifies the registration process with Detectron2. We plan to provide conversion scripts from other common formats in the near future. 

### Dependencies

### Installation


### Usage
