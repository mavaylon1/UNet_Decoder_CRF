#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, ZeroPadding2D,     Dropout, Conv2DTranspose, Cropping2D, Add, UpSampling2D, BatchNormalization
from keras.layers.merge import concatenate
from image_segmentation_keras.keras_segmentation.models.model_utils import get_segmentation_model
from glob import glob

import tensorflow as tf

strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

i=0
# Open a strategy scope.
if i==0:
    input_height = 512
    input_width = 512
    n_classes = 2
    channels = 3

    img_input = Input(shape=(input_height,input_width, channels))

    conv0 = Conv2D(64, (3, 3), activation='relu', padding='same')(img_input)
#     conv0 = Dropout(0.2)(conv0)
    conv0 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv0)
    bn0 = BatchNormalization()(conv0)
    pool0 = MaxPooling2D((2, 2))(bn0)
    
    conv1 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool0)
#     conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv1)
    bn1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D((2, 2))(bn1)

    conv2 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool1)
#     conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv2)
    bn2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D((2, 2))(bn2)

    conv3 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool2)
#     conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv3)
    bn3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D((2, 2))(bn3)
    
    conv4 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool3)
#     conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv4)
    print("conv4",conv4.shape)
    print('conv3',conv3.shape)

    up_= Conv2DTranspose(512,(2,2),strides=2,padding='same')(conv4)
    print('up_',up_.shape)
    up0 = concatenate([up_, conv3], axis=3)
    print(up0.shape)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(up0)
#     conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    bn4 = BatchNormalization()(conv5)
    
    up_2= Conv2DTranspose(256,(2,2),strides=2,padding='same')(bn4)
    up1 = concatenate([up_2, conv2], axis=-1)
    print(up1.shape)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up1)
#     conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
    bn5 = BatchNormalization()(conv6)
    
    up_3= Conv2DTranspose(128,(2,2),strides=2,padding='same')(bn5)
    up2 = concatenate([up_3, conv1], axis=3)
    print(up2.shape)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up2)
#     conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
    bn6 = BatchNormalization()(conv7)
    
    up_4= Conv2DTranspose(64,(2,2),strides=2,padding='same')(bn6)
    up3 = concatenate([up_4, conv0], axis=3)
    print(up3.shape)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up3)
#     conv8 = Dropout(0.2)(conv8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
    bn7 = BatchNormalization()(conv8)
    print(conv8.shape)
    out = Conv2D( n_classes, (1, 1) , padding='same')(bn7)
    print('out',out.shape)
    model = get_segmentation_model(img_input ,  out ) # this would build the segmentation model


# In[2]:

model.load_weights('/home/maavaylon/LBNL_Seg/Weights/unet_vanilla_davis.h5')

model.evaluate_segmentation(inp_images_dir='//home/maavaylon/Unet_Vanilla_Davis_Results', annotations_dir='/home/maavaylon/proximal_lower_contrast_test/ann')




