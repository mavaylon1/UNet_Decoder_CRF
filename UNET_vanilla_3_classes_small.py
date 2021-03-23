#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.layers import Conv2D, MaxPooling2D, Input, ZeroPadding2D, Input, Dropout, Conv2DTranspose, Cropping2D, Add, UpSampling2D, BatchNormalization, Activation
from keras.models import Model
from keras.layers.merge import concatenate
from image_segmentation_keras.keras_segmentation.models.model_utils import get_segmentation_model
from glob import glob


# In[2]:


def conv_block(inputs, filters, pool=True):
    x = Conv2D(filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    if pool == True:
        p = MaxPooling2D((2, 2))(x)
        return x, p
    else:
        return x
def build_unet(shape, num_classes):
    img_input = Input(shape)

    """ Encoder """
    x1, p1 = conv_block(img_input, 64, pool=True)
    x2, p2 = conv_block(p1, 128, pool=True)
    x3, p3 = conv_block(p2, 256, pool=True)
    x4, p4 = conv_block(p3, 512, pool=True)

    """ Bridge """
    b1 = conv_block(p4, 1024, pool=False)

    """ Decoder """
    u1 = UpSampling2D((2, 2), interpolation="bilinear")(b1)
    c1 = concatenate([u1, x4],axis=3)
    x5 = conv_block(c1, 512, pool=False)

    u2 = UpSampling2D((2, 2), interpolation="bilinear")(x5)
    c2 = concatenate([u2, x3],axis=3)
    x6 = conv_block(c2, 256, pool=False)

    u3 = UpSampling2D((2, 2), interpolation="bilinear")(x6)
    c3 = concatenate([u3, x2],axis=3)
    x7 = conv_block(c3, 128, pool=False)

    u4 = UpSampling2D((2, 2), interpolation="bilinear")(x7)
    c4 = concatenate([u4, x1],axis=3)
    x8 = conv_block(c4, 64, pool=False)

    """ Output layer """
    output = Conv2D(num_classes, 1, padding="same", activation="softmax")(x8)

    return get_segmentation_model(img_input ,  output )


# In[3]:


shape = (256, 256, 3)
num_classes = 3
model = build_unet(shape, num_classes)
model.summary()


# In[4]:


#shuffled
model.train(
    train_images =  "/home/maavaylon/Data1/train/img/",
    train_annotations = "/home/maavaylon/Data1/train/ann/",
    epochs=20,
    steps_per_epoch=len(glob("/home/maavaylon/Data1/train/img/*")),
    batch_size=1,
    validate=True,
    val_images="/home/maavaylon/Data1/test/img/",
    val_annotations="/home/maavaylon/Data1/test/ann/",
    val_batch_size=1,
    val_steps_per_epoch=len(glob("/home/maavaylon/Data1/test/img/*"))
)


# In[ ]:


