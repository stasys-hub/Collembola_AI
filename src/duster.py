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
import PIL
import random
import rglob
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense

PIL.Image.MAX_IMAGE_PIXELS = 500000000

def dump_training_set(train_directory, dust_directory, duster_path, df_train, df_dust):
    '''
    Creates the duster training set by extracting subpictures of animals and else (background and dust) from
    the mesofauna community detection training set.
    Subpictures are saved as jpeg in the 'duster' directory tree
    '''
    
    # Extracting animal subpictures
    for file in df_train.file_name.unique():
        im = Image.open(train_directory + '/' + file)
        for raw in df_train[df_train['file_name']==file][['box', 'id']].values:
            
            os.makedirs(f'{duster_path}/train/Animal', exist_ok=True)
            im.crop(raw[0].bounds).save(f'{duster_path}/train/{raw[1]}.jpg','JPEG')        
        im.close()
        
    # Extracting dust subpictures
    for file in df_dust.file_name.unique():
        im = Image.open(dust_directory + '/' + file)
        for raw in df_dust[df_dust['file_name']==file][['box', 'id']].values:
            os.makedirs(f'{duster_path}/train/Dust', exist_ok=True)
            im.crop(raw[0].bounds).save(f'{duster_path}/dust/{raw[1]}.jpg','JPEG')        
        im.close()           

    # Reserving 20% of the animal and dust subpictures for model validation
    animals_list= list(Path(f'{duster_path}/train').rglob('*.jpg'))    
    random.shuffle(animals_list)
    num_val = len(animals_list) * 20 / 100
    os.makedirs(f'{duster_path}/validation/Animal', exist_ok=True)
    for i in animals_list[:num_val]:
        shutil.move(i, f'{duster_path}/validation/Animal/{ntpath.basename(i)}')
        
    dust_list= list(Path(f'{duster_path}/dust').rglob('*.jpg'))        
    random.shuffle(dust_list)
    num_val = len(dust_list) * 20 / 100
    os.makedirs(f'{duster_path}/validation/Dust', exist_ok=True)
    for i in dust_list[:num_val]:
        shutil.move(i, f'{duster_path}/validation/Dust/{ntpath.basename(i)}')    

    
def get_image_datagen():
    
    rescale=1./255   
    train_datagen = ImageDataGenerator(
        rescalerescale,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

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


def train_duster(duster_path):
    
    batch_size = 16
    train_datagen, test_datagen = get_image_datagen()

    train_generator = train_datagen.flow_from_directory(
        f'{duster_path}/train',  
        classes=['Animal','Dust'],
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')
    
    validation_generator = test_datagen.flow_from_directory(
        f'{duster_path}/validation',
        classes=['Animal','Dust'],
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')
    
    model = model_configure()    
    model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)
    
    model.save_weights(f'{duster_path}/dust.h5')
    
def load_duster_and_classify(model_weights, duster_path):
    '''
    The trained duster will classifiy the pictures in the duster_path in the subfolder 'to_predict'
    Results are written in the dust.csv
    '''
    
    model = model_configure()
    model.load_weights(f'{duster_path}/dust.h5')
    
    train_datagen, test_datagen = get_image_datagen()
    
    test_generator = test_datagen.flow_from_directory(
        duster_path,
        classes=['unknown'],
        class_mode=None,
        shuffle=False,
        target_size=(150, 150),
        batch_size=batch_size)
    
    y_pred = [1 if p[0] > 0.5 else 0 for p in y ]
    class_indices = {0: 'Animal', 1: 'Dust'}
    y_class = [class_indices[i] for i in y_pred]
    labels_files = list(zip(y_class, test_generator.filenames))

    df = pd.DataFrame(labels_files, columns=['dust', 'id'])
    df['id'] = df.id.str.replace('unknown/', '').str.replace('.jpg', '')
    
    df.to_csv(f'{duster_path}/dust.csv', index=False)
    
    return df
