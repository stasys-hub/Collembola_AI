![Screenshot](CAI_git.png)
## Welcome to Collembola_AI
in this project we retrained the popular [Faster R-CNN](https://arxiv.org/pdf/1506.01497.pdf) Convolutional Neural Network using Facebooks [Detectron2](https://github.com/facebookresearch/detectron2) Framework to detect and classify species in a mock community of soil mesofauna preserved in ethanol. The community is made of 10 Collembola species and 2 Acari species, and were acquired using our homemade macrophotography workflow "HoverMacroCam" (described in Sys et al (submitted), arduino control scripts provided here, contact us for details in the mean time).

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


We provide a wrapper script, collembolAI.py, written in python to perform the inference workflow on ones dataset with only having to change a few lines of code. Furthermore this wrapper script simplifies the registration of custom datasets. Since we provide our own trained model you can retrain this model to extend it's classes and usability. 

Currently we expect the labeled image data to be labeled in COCO format as this immensely simplifies the registration process with Detectron2. For our own labeling workflow we used [labelImg](https://github.com/tzutalin/labelImg) with PascalVOC labels format. The conversion to COCO format can be done with [voc2coco](https://github.com/yukkyo/voc2coco), a version of the script is distributed in this project.

We wrote the code on a machine running Ubuntu 20.04 LTS, but any Linux environment allowing you to run Pytorch with proprietary Nvidia drivers (tested with 450.102.04) and CUDA (tested with 11.0/1 & 10.2) should work. MacOS and Windows could also work if you get Detectron running, but no guarantees on that. The VRAM you will need, will depend on the model you take from the [Model Zoo](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md). We were using the Faster R-CNN R50-FPN model on a RTX 2060 Super GPU without any hassles.

### Dependencies

1. Python 3 (a few packages found in [requirements.txt](src/requirements.txt))
2. Detectron 2 - since we use the Detectron2 API we have the same Dependecies. Please refer to their [Documentation](https://detectron2.readthedocs.io/tutorials/install.html#requirements). 
3. CUDA enabled Nvidia GPU -> checkout the [compatibility list ](https://detectron2.readthedocs.io/en/latest/tutorials/install.html#install-pre-built-detectron2-linux-only) of detectron and CUDA (We ran it on a RTX 2060 Super, Nvidia Titan X and V100)

### Installation

Suggestion on how to install Detectron2 with CAI:

- replace values in [ ] according to your system


```bash
# create a python env with python 3 (tested with 3.8+)
python -m venv [path/to/env]

# activate env
source activate [path/to/env]/bin/activate

# install pytorch and torchvision for e.g. CUDA 11.1 (may differ -> depends on your setup*) 
pip3 install torch==1.10.0+cu111 torchvision==0.11.1+cu111\
 torchaudio==0.10.0+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html

# install Detectron2 
python -m pip install detectron2 -f\
 https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html

# install other deps from requirements.txt
pip3 install -r requirements.txt
```
*for CUDA installation support please refer to the [nvidia docs](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)


To use CollembolAI just clone the git Directory into a place of your choice and follow the [installation](#installation) if not already happened. 

### Usage
The Script collembolAI.py shipped with CollembolAI is the heart of the pipeline. The simplest way to get started on an already annotated dataset is:

1. to create a project directory (e.g. /home/clem/MyComputerVisionProject), a train directory (/home/clem/MyComputerVisionProject/train) and a test directory (/home/clem/MyComputerVisionProject/test) and optionnaly a dust directory (/home/clem/MyComputerVisionProject/dust)
2. In the train directory, put the training pictures with their annotation file in COCO format (must be named train.json)
3. In the test directory, put the test pictures with their annotation file in COCO format (must be named test.json)
4. (optional) In the dust directory, put some background-only pictures (no organism on them).
4. Create a new template (its a simple text file) to set up the retraining parameters of a model and give this training run a name. You can re-use and adapt the template provided with CollembolAI [template.conf](src/template.conf).
5. In the template, make sure to indicate the path to you project folder, and give a name to your model
6. The template default to the faster rcnn R50 FPN 3x model, that should work on most modern laptop equipped with a good NVIDIA GPU ~ 8GB VRAM). To select another base model provided by Detectron2, please check the [model zoo](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md) and change "model_zoo_config" in the configuration file accordingly. The  Faster RCNN models should work, provided your hardware can follow.
7. If you want to perform some automatic annotation on new pictures using your model on new pictures, drop them in a new folder (e.g. 'inference/') and indicate this folder name in the configuration file using 'inference_directory'.
8. For the other optional configuration variable, please check Detectron2 documentation


#### Example of configuration file
```
[DEFAULT]
project_directory = /home/clem/MyComputerVisionProject
model_name = mymodel_using_faster_rcnn_R50_FPN
 
[OPTIONAL]
inference_directory = inference
model_zoo_config = COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml
iterations = 8000
number_of_workers = 2
batch_size = 5
learning_rate = 0.00025 
detection_treshold = 0.6
gpu_device_num = 0
duster = True
```

#### Running the pipeline

1. describe the training set
```bash
python3 collembolAI.py -s template.conf
```
<br/>
2. training the detector/classificator model (SAHI/Detectron2)
```bash
python3 collembolAI.py -t template.conf
```
<br/>
3. training the duster (optional): a CNN model to recognize false positive (typically dirt, dust, background specks) in the detector output. Depends on the above step, and needs some background only pictures in the 'dust' folder. If duster = True in configuration file, the duster will be already trained in step 2.
```bash
python3 collembolAI.py -d template.conf
```
<br/>
3. testing: evaluate the model performance on the test set. If duster = True in configuration file, the duster will be used to further clean the output.
```bash
python3 collembolAI.py -e template.conf
```
<br/>
4. annotate new pictures: perform inferences on new pictures found in the inference directory. If duster = True in configuration file, the duster will be used to further clean the output.
```bash
python3 collembolAI.py -a template.conf
```
<br/>
5. Running the whole pipeline: training, testing, annotating in one go.
```bash
python3 collembolAI.py -t -e -a template.conf
```

### Overview of folders organisation 
- In black: names and folder organisation are at the user discretion; <span style="color:blue">in blue: organisation is constrained (= files must be there but name them as you like)</span>; <span style="color:red">in red: organisation and naming are constrained.</span>

<pre>
    Collembola_AI/ .............. This repository
          `src/ ...................... Scripts and python modules.

    MyComputerVisionProject/ ... a subdirectory for a given project.
          |--DataFactory/ ............ In which the dataset is being build (stitching pictures, annotations)
          |--Datasets/ ............... In which finalized datasets are kept
          |--Model/ .................. In which a model is going to be build, on a given training and test set
              <span style="color:red">|--train/ .............. The training set for this model (selected from Datasets)
              <span style="color:blue">|   |--train_01.jpg
              |   |--train_02.jpg
              |   |--[...]
              |   |--train_n.jpg</span>
              |   |--train.json...... Annotations of the train set, in COCO format (json)
              |--test/ .............. The test set for this model (selected from Datasets)
              <span style="color:blue">|   |--test_01.jpg
              |   |--test_02.jpg
              |   |--[...]
              |   |--test_n.jpg</span>
              |   |--test.json ..... Annotations of the test set, in COCO format (json)
              |--dust/ ............. The 'dust' set for training the 'duster' (= unannotated pictures of background, without organisms)
             <span style="color:blue"> |   |--dust_01.jpg
              |   |--dust_02.jpg
              |   |--[...]
              |   |--dust_n.jpg</span>
              |--duster/ .......... In which the duster will be build</span>
              <span style="color:blue">
              |--inference/ ....... In which you can put some pictures you want to automatically annotate using the model.
              `--MyModel.conf ..... The model configuration file</span>
</pre>


### HoverMacroCam
You reproduced our macrophography system ? Great, you can find some arduino scripts in the "hovermacro_control" folder of the repo. canon_apsc.ino was suitable for using with a Canon EOS 7D DSLR. pentax_ff.ino is our running version using a Pentax k1 II DSLR. The program has to be adapted if using a new brand or a different sensor size / resolution. Get in touch if you need help. Note that you can adjust the surface to cover to your needs and produce images of various size.


### Building a dataset

#### Acquire pictures
CollembolAI is designed to be applied on large to very large pictures including a large number of small objects. While our object of study is soil fauna, the pipeline would apply to any comparable problem. To follow our steps, first is to acquire the collection of pictures. If you are using HoverMacroCam, make sure that your background is visible and showing a random pattern to guide the stitching. In our case, the paintbrush strokes did the job. Note: we are now working on an approach that do not rely on image stitching anymore and will simplify the image acquition. You are welcome to get in touch with us.

Stitching: gather all the pictures that belong to the same composite in a directory. When preparing several composite, we recommend the following directory organisation:
<pre>
    DataFactory/ ................... the root folder in which the dataset is being build.
     `--Project/ ................... a subdirectory for a given project.
          |--to_stitch/ ............ gather here the pictures to stitch
          |   |-- set01/ ........... pictures for the first composite, folder's name is at the user discretion
          |   |   |
          |   |   |-- 01.jpg
          |   |   |-- 02.jpg
          |   |   |-- [...]
          |   |   |-- 04.jpg
          |   |-- set02/ 
          |   |-- [...]/
          |   `-- setn/
          `-- stitched/ ............ the output directory, where the stitched composites will be written
</pre>
All the folder and file names above are at the user discretion. Mind that the pictures should have the jpg, JPG or JPEG extension.

Run
```bash
stitch_macro.py -o sitched to_stitch
```
<br/>
In the "done" folder, the stiched composite will be written as: set01.jpg, set02.jpg, [...], set03.jpg (file name follows folders name). The folders inside 'to_stitch' will also be moved to 'done/' in order to backup the initial pictures.

For detail run:
```bash
stitch_macro.py -h
```
<br/>

#### Annotate pictures.


Our pipeline support annotations using [labelImg](https://github.com/tzutalin/labelImg) using XML PascalVOC labels format that will be converted to JSON COCO format.
Of course, you can choose the software of your liking. In the end, <strong>you need to produce annotation in the JSON COCO format.</strong>
<br/>

##### 1. The normal way.
Using labelImg the 'normal way', simply use the software to annotate each specimens in your pictures, savin annotations in PascalVOC format. Then convert those annotations in a single COCO JSON file using the third party script [voc2coco](https://github.com/yukkyo/voc2coco) (we provide a version of it in this repository though).
One way to use voc2coco.py is to list the labels names in a text file (e.g. labels.txt) and the PascalVOC files path in another file (e.g. annotations_paths.txt), then run:
```bash
voc2coco.py --ann_paths_list annotations_paths.txt --labels labels.txt --output dataset.json
```
e.g. annotations_paths.txt:
  <pre><code>
   ~/DataFactory/Project/stitched/set01.xml
   ~/DataFactory/Project/stitched/set02.xml
   ~/DataFactory/Project/stitched/set03.xml
   [...]</code></pre>

e.g. labels.txt:
  <pre><code>
   speciesname01
   speciesname02
   speciesname03
   [...]</code></pre>

Once you have your COCO JSON file (here dataset.json), you can run our sanitization script, that will tidy the file (such as re-indexing and adding some fields that may pose issue later if missing). The script will output cleaned copy of this file with the 'sanitized' suffix (e.g. dataset.json.sanitized).

```bash
sanitize_cocofromvoc.py ./dataset.json
```

(Note that this script can also be used to drop the annotations belonging to a given category (objects belonging to this categorie will no longer be annotated). For example the following command will drop the categorie which id is 0.
```bash
sanitize_cocofromvoc.py --drop_cat 0 -c ./dataset.json.reviewed
```
)

Once your are done, your working folder will look like this:

<pre>
    DataFactory/ ................... the root folder in which the dataset is being build.
     `- -Project/ .................. a subdirectory for a given project.
          `-- stitched/ ............ the output directory, where the stitched composites are written
              |-- set01/ ........... backup of the pictures
              |-- set01.jpg ........ large composite picture
              |-- set01.xml ........ PascalVOC annotation file
              |-- set02/
              |-- set02.jpg
              |-- set02.xml
              |-- set03/
              |-- set03.jpg
              |-- set03.xml
              |-- [...]
              |-- setn/
              |-- setn.jpg
              |-- setn.xml
              |-- annotations_paths.txt
              |-- labels.txt
              |-- dataset.json
              <span style="color:red"> |-- dataset.json.sanitized</span>
</pre>             

To train the model you need only the composite pictures (jpg) and the final COCO json file (dataset.json.sanitized). We recommend to keep the other files until you are sure you don't need them anymore.
 

##### 2. Our alternative process

Here is a hint to speed up annotation by introducing a bit of "taylorism" (note that this is a matter of work style. If you are unsure, then using labelImg for attributing all the labels may work better for you):

1. Use labelImg to draw the bounding box around all the specimens without worrying about attributing the correct label (attribute 'collembola' to all of them for example). Thus, any untrained person can help with this task.

2. Convert the annotations to COCO JSON format using voc2coco.py (as instructed above).
```bash
voc2coco.py --ann_paths_list annotations_paths.txt --labels labels.txt --output dataset.json
```

4. Prepare your list of labels in json format following COCO categories structure. Example:

<pre><code>
    "categories": [
        {
            "supercategory": "none",
            "id": 0,
            "name": "Sminthurides_aquaticus__281415"
        },
        {
            "supercategory": "none",
            "id": 1,
            "name": "Folsomia_candida__158441"
        },
        {
            "supercategory": "none",
            "id": 2,
            "name": "Lepidocyrtus_lignorum__707889"
        }
    ]
</code></pre>

Edit the dataset.json file with a text editor and replace the pre-existing "categories" block with the new one. Save it.
    
5. Attribute labels to each specimens by running:    
```bash
review_coco_annotations.py dataset.json
```
<br/>
The first specimen will be displayed in a popup, and you will be asked to enter the correct label ID in the terminal prompt. Once done, press enter and the next specimen will show up. If you made an error and already validated it, simply interrupt the script, open dataset.json.reviewed, find the last annotation block (at the end of the file) and correct manually the categorie_id then save the file. Run the script again, it will resume. You can interrupt the script at any time, running it again will resume wher you left it.


6. Run the sanitization script (as instructed above) on the reviewed file.

```bash
sanitize_cocofromvoc.py ./dataset.json.reviewed
```


7. Split dataset in a train / test set.
    
You may repeat the above steps in a second time to prepare a test set, or simply split the dataset at this stage in a training set and a test set.


To select at random a given percentage of pictures to use as test, use the command:
```bash
cocosets_utils.py --split dataset.json.reviewed --ratio 20
```
<br/>
it will move 20% of the pictures selected at random in a child 'test' folder and write a test.json COCO file, the remaining pictures are moved in a child 'train' folder along with a train.json COCO file.

<br/>
Alternatively, you can create the train and test folder yourself and dispatch the pictures manually. Then run the following command to obtain the reduced COCO file:
```bash
cocosets_utils.py --split ./dataset.json.reviewed
```

At this point you can start to use CollembolAI.py as described in <strong>Usage</strong>