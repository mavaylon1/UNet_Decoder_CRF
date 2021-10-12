from keras.layers import *
from keras.layers.merge import concatenate
from UNet_Decoder_CRF.keras_segmentation import train
from glob import glob
from UNet_Decoder_CRF.keras_segmentation.models.config import IMAGE_ORDERING
from UNet_Decoder_CRF.keras_segmentation.models.model_utils import get_segmentation_model

channels, height, width = 3, 256, 256
n_classes = 3


# In[3]:


def one_side_pad(x):
    x = ZeroPadding2D((1, 1))(x)
    x = Lambda(lambda x: x[:, :-1, :-1, :])(x)
    return x


# In[4]:


# Input
input_shape = (height, width, 3)
img_input = Input(shape=input_shape)
print(img_input.shape)
# Add plenty of zero padding
x = ZeroPadding2D(padding=(100, 100))(img_input)

# VGG-16 convolution block 1
x = Conv2D(64, (3, 3), activation='relu', padding='valid', name='conv1_1')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

# VGG-16 convolution block 2
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2', padding='same')(x)

# VGG-16 convolution block 3
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3', padding='same')(x)
pool3 = x

# VGG-16 convolution block 4
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4', padding='same')(x)
pool4 = x

# VGG-16 convolution block 5
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='pool5', padding='same')(x)

# Fully-connected layers converted to convolution layers
x = Conv2D(4096, (7, 7), activation='relu', padding='valid', name='fc6')(x)
x = Dropout(0.5)(x)
x = Conv2D(4096, (1, 1), activation='relu', padding='valid', name='fc7')(x)
x = Dropout(0.5)(x)
x = Conv2D(n_classes, (1, 1), padding='valid', name='score-fr')(x)

# Deconvolution
score2 = Conv2DTranspose(n_classes, (4, 4), strides=2, name='score2')(x)
print(score2.shape)
# Skip connections from pool4
score_pool4 = Conv2D(n_classes, (1, 1), name='score-pool4')(pool4)
score_pool4c = Cropping2D((5, 5))(score_pool4)
score_pool4c = one_side_pad(score_pool4c)
score_fused = Add()([score2, score_pool4c])
score4 = Conv2DTranspose(n_classes, (4, 4), strides=2, name='score4', use_bias=False)(score_fused)
print(score4.shape)

# Skip connections from pool3
score_pool3 = Conv2D(n_classes, (1, 1), name='score-pool3')(pool3)
score_pool3c = Cropping2D((8, 8))(score_pool3)
score_pool3c = one_side_pad(score_pool3c)

# Fuse things together
score_final = Add()([score4, score_pool3c])

# Final up-sampling and cropping
upsample = Conv2DTranspose(n_classes, (16, 16), strides=8, name='upsample', use_bias=False)(score_final)
upscore = Cropping2D(((44, 44), (44, 44)))(upsample)

model= get_segmentation_model(img_input, upscore)

# model.train(
#     train_images =  "/home/maavaylon/Data1/train/img/",
#     train_annotations = "/home/maavaylon/Data1/train/ann/",
#     epochs=20,
#     steps_per_epoch=len(glob("/home/maavaylon/Data1/train/img/*")),
#     batch_size=1,
#     validate=True,
#     val_images="/home/maavaylon/Data1/test/img/",
#     val_annotations="/home/maavaylon/Data1/test/ann/",
#     val_batch_size=1,
#     val_steps_per_epoch=len(glob("/home/maavaylon/Data1/test/img/*"))
# )
