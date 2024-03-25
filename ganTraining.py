#Training GAN for generating synthetic Chest X-Ray Images


import os
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

images = np.array(images)
labels = np.array(labels)

images = images.astype(np.float32)

images /= 255.0

images = images.reshape((images.shape[0], images.shape[1], images.shape[2], 1))

print(images.shape)


D = Sequential()
depth = 64
dropout = 0.3
# In: 28 x 28 x 1, depth = 1
# Out: 14 x 14 x 1, depth=64
#input_shape = (self.img_rows, self.img_cols, self.channel)
input_shape = (256, 256, 1)

D.add(Conv2D(depth*1, 5, strides=2, input_shape=input_shape, padding='same', activation=LeakyReLU(alpha=0.2)))
D.add(Dropout(dropout))

D.add(Conv2D(depth*2, 5, strides=2, padding='same', activation=LeakyReLU(alpha=0.2)))
D.add(Dropout(dropout))


D.add(Conv2D(depth*4, 5, strides=2, padding='same', activation=LeakyReLU(alpha=0.2)))
D.add(Dropout(dropout))


D.add(Conv2D(depth*8, 5, strides=2, padding='same', activation=LeakyReLU(alpha=0.2)))
D.add(Dropout(dropout))

D.add(Conv2D(depth*8, 5, strides=2, padding='same', activation=LeakyReLU(alpha=0.2)))
D.add(Dropout(dropout))

D.add(Conv2D(depth*8, 5, strides=2, padding='same', activation=LeakyReLU(alpha=0.2)))
D.add(Dropout(dropout))
# Out: 1-dim probability
D.add(Flatten())
D.add(Dense(1))
D.add(Activation('sigmoid'))
D.summary()

G = Sequential()
dropout = 0.3
depth = 64+64+64+64
dim = 8
# In: 100
# Out: dim x dim x depth
G.add(Dense(dim*dim*depth, input_dim=100))
G.add(BatchNormalization(momentum=0.9))
G.add(Activation('relu'))
G.add(Reshape((dim, dim, depth)))
G.add(Dropout(dropout))
# In: dim x dim x depth
# Out: 2*dim x 2*dim x depth/2
#G.add(UpSampling2D())
G.add(Conv2DTranspose(int(depth/2), 5, strides=(2,2), padding='same'))
G.add(BatchNormalization(momentum=0.9))
G.add(Activation('relu'))

#G.add(UpSampling2D())
G.add(Conv2DTranspose(int(depth/4), 5, strides=(2,2), padding='same'))
G.add(BatchNormalization(momentum=0.9))
G.add(Activation('relu'))

#G.add(UpSampling2D())
G.add(Conv2DTranspose(int(depth/4), 5, strides=(2,2), padding='same'))
G.add(BatchNormalization(momentum=0.9))
G.add(Activation('relu'))

#G.add(UpSampling2D())
G.add(Conv2DTranspose(int(depth/8), 5, strides=(2,2), padding='same'))
G.add(BatchNormalization(momentum=0.9))
G.add(Activation('relu'))

#G.add(UpSampling2D())
G.add(Conv2DTranspose(int(depth/8), 5, strides=(2,2), padding='same'))
G.add(BatchNormalization(momentum=0.9))
G.add(Activation('relu'))

# Out: 28 x 28 x 1 grayscale image [0.0,1.0] per pix
G.add(Conv2DTranspose(1, 5, padding='same'))
G.add(Activation('sigmoid'))
G.summary()

optimizer = RMSprop(learning_rate=0.0007, clipvalue=1.0, decay=6e-8)
DM = Sequential()
DM.add(D)
DM.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['accuracy'])

optimizer = RMSprop(learning_rate=0.0003, clipvalue=1.0, decay=3e-8)
AM = Sequential()
AM.add(G)
AM.add(D)
AM.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['accuracy'])

for i in range(28001):
    batch_size = 64
    images_train = images[np.random.randint(0, images.shape[0], size=batch_size), :, :, :]

    noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
    images_fake = G.predict(noise)

    print(images_train.shape, images_fake.shape)
    x = np.concatenate((images_train, images_fake))
    y = np.ones([2*batch_size, 1])
    y[batch_size:, :] = 0

    d_loss = DM.train_on_batch(x, y)
    y = np.ones([batch_size, 1])
    noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
    a_loss = AM.train_on_batch(noise, y)
    
    if (i%2000)==0:
        if i==0:
            continue
        G.save(('./models/G'+str(i)))
        DM.save(('./models/DM'+str(i)))
        AM.save(('./models/AM'+str(i)))
        print(i)
    if (i%50)==0:
        print('******')

