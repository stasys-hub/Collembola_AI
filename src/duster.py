#!/usr/bin/env python3
"""
Project title:       CollembolAI
Authors:             Stephan Weißbach, Stanislav Sys, Clément Schneider
Original repository: https://github.com/stasys-hub/Collembola_AI.git
Module title:        duster.py
Purpose:             Identification of dust and background specks among the predicted labels (= removal of false positive)
Dependencies:        See ReadMe
Last Update:         31.01.2021
Licence:             

The following code is adapted from https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
"""
import os
import ntpath
from path import Path
import pandas as pd
import PIL
import random
import rglob
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from cocosets_utils import coco2df, crop_annotations

PIL.Image.MAX_IMAGE_PIXELS = 500000000

def dump_training_set(train_directory, dust_directory, duster_path, df_train, df_dust):
    '''
    Creates the duster training set by extracting subpictures of animals and else (background and dust) from
    the mesofauna community detection training set.
    Subpictures are saved as jpeg in the 'duster' directory tree
    '''
    
    # Write train subpictures from each animal classes
    def wipe_dir(path):
        if os.path.exists(path) and os.path.isdir(path):
            shutil.rmtree(path)
            
    duster_train_dir = os.path.join(duster_path, 'train')
    duster_train_all_dir = os.path.join(duster_train_dir, 'All')
    duster_val_dir = os.path.join(duster_path, 'validation')
    duster_val_all_dir = os.path.join(duster_val_dir, 'All')

    wipe_dir(duster_train_dir)
    crop_annotations(df_train, train_directory, duster_train_dir)

    #with open(os.path.join(dust_directory, 'inference.json'), 'r') as j:
    #    dust = coco2df(json.load(j))
    # df_dust['name'] = 'Dust'

    crop_annotations(df_dust, dust_directory, duster_train_dir)
        
    wipe_dir(duster_val_dir)

    os.makedirs(duster_train_all_dir, exist_ok=True)
    os.makedirs(duster_val_all_dir, exist_ok=True)

    for name in list(df_train['name'].unique()) + ['Dust']:

        class_file_list = list(Path(os.path.join(duster_train_dir, name)).rglob('*.jpg'))
        random.shuffle(class_file_list)
        num_val = int(len(class_file_list) * 20 / 100)
        os.makedirs(os.path.join(duster_val_dir, name), exist_ok=True)
        for i in class_file_list[:num_val]:
            if name != 'Dust':
                shutil.copy(i, os.path.join(duster_val_dir, f'{name}/{ntpath.basename(i)}'))
                shutil.move(i, os.path.join(duster_val_all_dir, ntpath.basename(i)))
            if name == 'Dust':
                shutil.move(i, os.path.join(duster_val_dir, f'{name}/{ntpath.basename(i)}'))
            
        if name != 'Dust':
            remains_class_file = list(Path(os.path.join(duster_train_dir, name)).rglob('*.jpg'))
            for i in remains_class_file:
                shutil.copy(i, os.path.join(duster_train_all_dir, ntpath.basename(i)))

                
def get_image_datagen():
    
    rescale=1./255   
    train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    test_datagen = ImageDataGenerator(rescale=rescale)
    
    return train_datagen, test_datagen

    
def model_configure():
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
    
    return model

def train_duster(duster_path, train_directory):
    
    classes = ['All'] + list(cocoj_get_categories(os.path.join(train_directory, 'train.json')).values())
    
    for ncls in classes:
    
        batch_size = 16
        train_datagen, test_datagen = get_image_datagen()

        train_generator = train_datagen.flow_from_directory(
            os.path.join(duster_path, 'train'),  
            classes=[ncls,'Dust'],
            target_size=(150, 150),
            batch_size=batch_size,
            class_mode='binary')
    
        validation_generator = test_datagen.flow_from_directory(
            os.path.join(duster_path, 'validation'),
            classes=[ncls,'Dust'],
            target_size=(150, 150),
            batch_size=batch_size,
            class_mode='binary')
    
        model = model_configure()    
        model.fit_generator(
            train_generator,
            steps_per_epoch=2000 // batch_size,
            epochs=50,
            validation_steps=800 // batch_size)
    
        model.save_weights(os.path.join(duster_path,f'{ncls}.h5'))
    
    
def load_duster_and_classify(duster_path, list_classes):
    '''
    The trained duster will classifiy the pictures in the duster_path in the subfolder 'to_predict'
    Results are written in the dust.csv
    '''
    
    results = pd.DataFrame()
    
    for ncls in list_classes:
    
        model = model_configure()
        model.load_weights(f'{duster_path}/{ncls}.h5')
    
        train_datagen, test_datagen = get_image_datagen()
    
        test_generator = test_datagen.flow_from_directory(
            f'{duster_path}/to_predict',
            classes=[ncls],
            class_mode=None,
            shuffle=False,
            target_size=(150, 150),
            batch_size=16)
    
        y = model.predict(test_generator, verbose=1)
    
        y_pred = [1 if p[0] > 0.5 else 0 for p in y ]
        class_indices = {0: ncls, 1: 'Dust'}
        y_class = [class_indices[i] for i in y_pred]
        labels_files = list(zip(y_class, test_generator.filenames))

        df = pd.DataFrame(labels_files, columns=['dust', 'id'])
        df['id'] = df.id.str.replace(f'{ncls}/', '').str.replace('.jpg', '')
        
        results = pd.concat([results, df], axis=0)
    
    results.to_csv(f'{duster_path}/dust.csv', index=False)
    
    return df