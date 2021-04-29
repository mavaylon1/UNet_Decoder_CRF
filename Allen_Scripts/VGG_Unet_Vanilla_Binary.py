#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.layers import *
from keras.models import Model
from keras import layers
from keras.layers.merge import concatenate
import sys
sys.path.insert(1, '../image_segmentation_keras')
from keras_segmentation.models.config import IMAGE_ORDERING

from keras_segmentation.models.model_utils import get_segmentation_model
from glob import glob


# In[ ]:


def unet_conv_block(inputs, filters, pool=True, batch_norm_first=True):
    if batch_norm_first == True:
        x = Conv2D(filters, 3, padding="same")(inputs)
#         x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2D(filters, 3, padding="same")(x)
#         x = BatchNormalization()(x)
        x = Activation("relu")(x)
    elif batch_norm_first == False:
        x = Conv2D(filters, 3, padding="same")(inputs)
        x = Activation("relu")(x)
#         x = BatchNormalization()(x)

        x = Conv2D(filters, 3, padding="same")(x)
        x = Activation("relu")(x)
#         x = BatchNormalization()(x)

    if pool == True:
        p = MaxPooling2D((2, 2))(x)
        return [x, p]
    else:
        return x


# In[4]:


def _unet(n_classes, encoder, l1_skip_conn=True, input_height=416,
          input_width=608):

  
    img_input, levels = encoder(
        input_height=input_height, input_width=input_width)
    [f1, f2, f3, f4, f5] = levels
    
    print("f5",f5.shape)

    o = f5
    
    """ Bridge """
    o = unet_conv_block(o, 512, pool=False)
    
    x = UpSampling2D((2, 2))(f5)
    x = concatenate([x, f4], axis=3)
    x = unet_conv_block(x, 512, pool=False, batch_norm_first=True)

    x = UpSampling2D((2, 2))(x)
    x = concatenate([x, f3], axis=3)
    x = unet_conv_block(x, 256, pool=False, batch_norm_first=True)

    x = UpSampling2D((2, 2))(x)
    x = concatenate([x, f2], axis=3)
    x = unet_conv_block(x, 128, pool=False, batch_norm_first=True)

    x = UpSampling2D((2, 2))(x)
    x = concatenate([x, f1], axis=3)
    x = unet_conv_block(x, 64, pool=False, batch_norm_first=True)

    x = Conv2D(n_classes, (3, 3), padding='same')(x)
    model = get_segmentation_model(img_input, o)

    return model


# In[5]:



if IMAGE_ORDERING == 'channels_first':
    MERGE_AXIS = 1
elif IMAGE_ORDERING == 'channels_last':
    MERGE_AXIS = -1
def get_vgg_encoder(input_height=224,  input_width=224, pretrained='imagenet'):

    assert input_height % 32 == 0
    assert input_width % 32 == 0

    if IMAGE_ORDERING == 'channels_first':
        img_input = Input(shape=(3, input_height, input_width))
    elif IMAGE_ORDERING == 'channels_last':
        img_input = Input(shape=(input_height, input_width, 3))

    x = Conv2D(64, (3, 3), activation='relu', padding='same',
               name='block1_conv1', data_format=IMAGE_ORDERING)(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same',
               name='block1_conv2', data_format=IMAGE_ORDERING)(x)
    p1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool',
                     data_format=IMAGE_ORDERING)(x)
    f1 = x
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same',
               name='block2_conv1', data_format=IMAGE_ORDERING)(p1)
    x = Conv2D(128, (3, 3), activation='relu', padding='same',
               name='block2_conv2', data_format=IMAGE_ORDERING)(x)
    p2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool',
                     data_format=IMAGE_ORDERING)(x)
    f2 = x

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               name='block3_conv1', data_format=IMAGE_ORDERING)(p2)
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               name='block3_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               name='block3_conv3', data_format=IMAGE_ORDERING)(x)
    p3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool',
                     data_format=IMAGE_ORDERING)(x)
    f3 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block4_conv1', data_format=IMAGE_ORDERING)(p3)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block4_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block4_conv3', data_format=IMAGE_ORDERING)(x)
    p4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool',
                     data_format=IMAGE_ORDERING)(x)
    f4 = x

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block5_conv1', data_format=IMAGE_ORDERING)(p4)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block5_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block5_conv3', data_format=IMAGE_ORDERING)(x)
    f5 = x

#     if pretrained == 'imagenet':
#         VGG_Weights_path = keras.utils.get_file(
#             pretrained_url.split("/")[-1], pretrained_url)
#         Model(img_input, x).load_weights(VGG_Weights_path)

    return img_input, [f1, f2, f3, f4, f5]


# In[6]:


def vgg_unet(n_classes, input_height=416, input_width=608, encoder_level=3):

    model = _unet(n_classes, get_vgg_encoder,
                  input_height=input_height, input_width=input_width)
    model.model_name = "vgg_unet"
    return model


# In[7]:


model = vgg_unet(n_classes=2,input_height=512, input_width=512)


# In[8]:


model.train(
    train_images = "/home/maavaylon/Binary_Data/BP_lrc_training/img/",
    train_annotations = "/home/maavaylon/Binary_Data/BP_lrc_training/ann/",
    epochs=20,
    steps_per_epoch=len(glob("/home/maavaylon/Binary_Data/BP_lrc_training/img/*")),
    batch_size=1,
    validate=True,
    val_images="/home/maavaylon/Binary_Data/BP_lrc_testing/img/",
    val_annotations="/home/maavaylon/Binary_Data/BP_lrc_testing/ann/",
    val_batch_size=1,
    val_steps_per_epoch=len(glob("/home/maavaylon/Binary_Data/BP_lrc_testing/img/*"))
)


# In[ ]:




