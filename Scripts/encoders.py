from model_utils import *
from keras.layers import Conv2D, MaxPooling2D, Input, ZeroPadding2D, \
    Dropout, Conv2DTranspose, Cropping2D, Add, UpSampling2D

def resnet50_encoder(input_height,  input_width):

    input_height = kwargs[input_height]
    input_width = kwargs[input_width]

    assert input_height % 32 == 0
    assert input_width % 32 == 0

    img_input = Input(shape=(input_height, input_width, 3))

    bn_axis = 3


    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7),
               strides=(2, 2), name='conv1')(x)
    f1 = x

    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = resnet50_conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = resnet50_identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = resnet50_identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    f2 = one_side_pad(x)

    x = resnet50_conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = resnet50_identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = resnet50_identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = resnet50_identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    f3 = x

    x = resnet50_conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = resnet50_identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = resnet50_identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = resnet50_identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = resnet50_identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = resnet50_identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    f4 = x

    x = resnet50_conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = resnet50_identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = resnet50_identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    f5 = x

    return img_input, [f1, f2, f3, f4, f5]

def unet_encoder(input_height, input_width, filters, depth, channels, batch_norm_first):

    input_height = kwargs[input_height]
    input_width = kwargs[input_width]
    filters = kwargs[filters]
    depth = kwargs[depth]
    channels = kwargs[channels]
    batch_norm_first = kwargs[batch_norm_first]
    pool = True

    assert input_height % 32 == 0
    assert input_width % 32 == 0

    x = Input(shape=(input_height, input_width, channels))

    blocks = []
    for i in range(depth):
        x = unet_conv_block(x, filters, pool, batch_norm_first)
        blocks += x
        if i < max(range(depth)):
            filters = filters*2

    return img_input, blocks

def vgg16_encoder(input_height,  input_width):

    input_height = kwargs[input_height]
    input_width = kwargs[input_width]

    assert input_height % 32 == 0
    assert input_width % 32 == 0

    img_input = Input(shape=(input_height, input_width, 3))

    x = Conv2D(64, (3, 3), activation='relu', padding='same',
               name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same',
               name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    f1 = x
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same',
               name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same',
               name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    f2 = x

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    f3 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    f4 = x

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    f5 = x

    return img_input, [f1, f2, f3, f4, f5]
