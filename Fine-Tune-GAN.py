#Fine Tuning GAN for chest x-ray image classification
#In this code we replace the last layer of discriminator
#and fine-tune it for classification


import os
os.environ["CUDA_VISIBLE_DEVICES"]="1" # second gpu
import cv2
import random
import numpy as np
import csv
from keras.layers import Dense, Flatten, Input, Conv2D, Conv2DTranspose,Dropout, Activation,LeakyReLU, BatchNormalization, Reshape, UpSampling2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def read_resize_gray_images(path):
    """
    Converts images to histograms for a given directory path.

    Args:
        path (str): Directory path containing images.

    Returns:
        tuple: A tuple containing two lists: histograms (hists) and labels (labels).
            - hists (list): A list of histograms, one for each image.
            - labels (list): A list of labels, indicating the class of each image.
    """
    # Initialize counters and lists
    class_iter = 0
    img_iter = 0
    images = []
    hists = []
    labels = []
    first = True

    # Traverse through the directory
    for dirname, _, filenames in os.walk(path):
        filesList = []
        # Skip the first iteration
        if first == True:
            first = False
            continue
        # Process each file in the directory
        for filename in filenames:
            filesList.append(os.path.join(dirname, filename)) 

            # Read and resize the image
            resized = cv2.resize(cv2.cvtColor(cv2.imread(os.path.join(dirname, filename)), cv2.COLOR_BGR2GRAY), (256,256), interpolation = cv2.INTER_AREA)

            #print(os.path.join(dirname, filename))
            # Append the resized image to the 'images' list
            images.append(resized)
            
            # Append the class label to the 'labels' list
            labels.append(class_iter)

            # Update image counter
            img_iter += 1
            
        # Update class counter
        class_iter += 1
        #if class_iter==2:
            #break
        
    # Return the lists of images and labels
    return images, labels



folder_path = './chest_xray/train/'

images, labels = read_resize_gray_images(folder_path)
print('Total Number of images:',len(images))  # prints the number of images read from the folder

folder_path = './chest_xray/test/'

test_images, test_labels = read_resize_gray_images(folder_path)
print('Total Number of test images:',len(test_images))  # prints the number of images read from the folder

test_images = np.array(test_images)
test_labels = np.array(test_labels)
test_images = test_images.reshape((test_images.shape[0], test_images.shape[1], test_images.shape[2], 1))
test_images = test_images.astype(np.float32)
test_images /= 255.0



images = np.array(images)
labels = np.array(labels)

images = images.astype(np.float32)
images /= 255.0

images = images.reshape((images.shape[0], images.shape[1], images.shape[2], 1))

print(images.shape)



from keras.models import load_model
G = load_model('./models2/G28000')
DM = load_model('./models2/DM28000')
AM = load_model('./models2/AM28000')


DM.layers[0].summary()

DM.layers[0].pop()
DM.layers[0].pop()

DM.layers[0].summary()

for layer in DM.layers[0].layers:
    layer.trainable = True
    
DM.layers[0].add(Dense(2, name='classifier'))
DM.layers[0].add(Activation('softmax'))

from keras.optimizers import Adam
optimizer = Adam(learning_rate=0.0001)
DM.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

from keras.utils import to_categorical

# Convert the training and test labels to categorical format
labels = to_categorical(labels)
test_labels = to_categorical(test_labels)

print('Labels\' shape after converting to categorical:')
print('Train labels:',labels.shape)
print('Test labels:',test_labels.shape)

# Train the model on the training data, and store the training history
history = DM.fit(x=images,
                          y=labels,
                          epochs=35,
                          batch_size=32,
                          validation_data=(test_images, test_labels),
                          shuffle=True)


DM.save('./finalmodel/DM')

