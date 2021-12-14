from . import unet_conv_block, resnet50_identity_block, resnet50_conv_block, one_side_pad, resnet34_identity_block, resnet34_conv_block

from keras.layers import Conv2D, MaxPooling2D, Input, ZeroPadding2D, \
    Dropout, Conv2DTranspose, Cropping2D, Add, UpSampling2D, BatchNormalization, Activation

def resnet34_encoder(**kwargs):
    input_height = kwargs['input_height']
    input_width = kwargs['input_width']
    channels = kwargs['channels']

    assert input_height % 32 == 0
    assert input_width % 32 == 0

    img_input = Input(shape=(input_height, input_width, channels))

    bn_axis = 3


    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    f1 = x
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = resnet34_conv_block(x, 3, [64, 64], stage=2, block='a', strides=(1, 1))
    x = resnet34_identity_block(x, 3, [64, 64], stage=2, block='b')
    x = resnet34_identity_block(x, 3, [64, 64], stage=2, block='c')
    f2 = one_side_pad(x)


    x = resnet34_conv_block(x, 3, [128, 128], stage=3, block='a')
    x = resnet34_identity_block(x, 3, [128, 128], stage=3, block='b')
    x = resnet34_identity_block(x, 3, [128, 128], stage=3, block='c')
    x = resnet34_identity_block(x, 3, [128, 128], stage=3, block='d')
    f3 = x

    x = resnet34_conv_block(x, 3, [256, 256], stage=4, block='a')
    x = resnet34_identity_block(x, 3, [256, 256], stage=4, block='b')
    x = resnet34_identity_block(x, 3, [256, 256], stage=4, block='c')
    x = resnet34_identity_block(x, 3, [256, 256], stage=4, block='d')
    x = resnet34_identity_block(x, 3, [256, 256], stage=4, block='e')
    x = resnet34_identity_block(x, 3, [256, 256], stage=4, block='f')
    f4 = x

    x = resnet34_conv_block(x, 3, [512, 512], stage=5, block='a')
    x = resnet34_identity_block(x, 3, [512, 512], stage=5, block='b')
    x = resnet34_identity_block(x, 3, [512, 512], stage=5, block='c')
    f5 = x

    return img_input, [f1, f2, f3, f4, f5]

def resnet50_encoder(**kwargs):

    input_height = kwargs['input_height']
    input_width = kwargs['input_width']
    channels = kwargs['channels']

    assert input_height % 32 == 0
    assert input_width % 32 == 0

    img_input = Input(shape=(input_height, input_width, channels))

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

def resnet101_encoder(**kwargs):

    input_height = kwargs['input_height']
    input_width = kwargs['input_width']
    channels = kwargs['channels']

    assert input_height % 32 == 0
    assert input_width % 32 == 0

    img_input = Input(shape=(input_height, input_width, channels))

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
    x = resnet50_identity_block(x, 3, [256, 256, 1024], stage=4, block='g')
    x = resnet50_identity_block(x, 3, [256, 256, 1024], stage=4, block='h')
    x = resnet50_identity_block(x, 3, [256, 256, 1024], stage=4, block='i')
    x = resnet50_identity_block(x, 3, [256, 256, 1024], stage=4, block='j')
    x = resnet50_identity_block(x, 3, [256, 256, 1024], stage=4, block='k')
    x = resnet50_identity_block(x, 3, [256, 256, 1024], stage=4, block='l')
    x = resnet50_identity_block(x, 3, [256, 256, 1024], stage=4, block='m')
    x = resnet50_identity_block(x, 3, [256, 256, 1024], stage=4, block='n')
    x = resnet50_identity_block(x, 3, [256, 256, 1024], stage=4, block='o')
    x = resnet50_identity_block(x, 3, [256, 256, 1024], stage=4, block='p')
    x = resnet50_identity_block(x, 3, [256, 256, 1024], stage=4, block='q')
    x = resnet50_identity_block(x, 3, [256, 256, 1024], stage=4, block='r')
    x = resnet50_identity_block(x, 3, [256, 256, 1024], stage=4, block='s')
    x = resnet50_identity_block(x, 3, [256, 256, 1024], stage=4, block='t')
    x = resnet50_identity_block(x, 3, [256, 256, 1024], stage=4, block='u')
    x = resnet50_identity_block(x, 3, [256, 256, 1024], stage=4, block='v')
    x = resnet50_identity_block(x, 3, [256, 256, 1024], stage=4, block='w')
    f4 = x

    x = resnet50_conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = resnet50_identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = resnet50_identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    f5 = x

    return img_input, [f1, f2, f3, f4, f5]


def unet_encoder(**kwargs):

    input_height = kwargs['input_height']
    input_width = kwargs['input_width']
    filters = kwargs['filters']
    channels = kwargs['channels']
    batch_norm_first = kwargs['batch_norm_first']

    assert input_height % 32 == 0
    assert input_width % 32 == 0

    pool = True
    filters = filters[::-1]
    depth = len(filters)

    img_input = Input(shape=(input_height,input_width, channels))

    concat_layers = []

    for i in range(depth):
        if i==0:
            x=img_input
        if i < max(range(depth)):
            x = unet_conv_block(x, filters[i], pool, batch_norm_first)
            concat_layers.append(x[0])
            x=x[1]
        else:
            x = unet_conv_block(x, filters[i], pool, batch_norm_first)
            concat_layers.extend(x)

    return img_input, concat_layers

def vgg16_encoder(**kwargs):

    input_height = kwargs['input_height']
    input_width = kwargs['input_width']
    channels = kwargs['channels']

    assert input_height % 32 == 0
    assert input_width % 32 == 0

    img_input = Input(shape=(input_height, input_width, channels))

    x = Conv2D(64, (3, 3), activation='relu', padding='same',
               name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same',
               name='block1_conv2')(x)
    p1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    f1 = x
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same',
               name='block2_conv1')(p1)
    x = Conv2D(128, (3, 3), activation='relu', padding='same',
               name='block2_conv2')(x)
    p2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    f2 = x

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               name='block3_conv1')(p2)
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               name='block3_conv3')(x)
    p3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    f3 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block4_conv1')(p3)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block4_conv3')(x)
    p4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    f4 = x

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block5_conv1')(p4)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block5_conv3')(x)
    p5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    f5 = x

    return img_input, [f1, f2, f3, f4, f5, p5]
