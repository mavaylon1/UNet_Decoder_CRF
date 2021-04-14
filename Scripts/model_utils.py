from keras.layers import Conv2D, MaxPooling2D, Input, ZeroPadding2D, \
    Dropout, Conv2DTranspose, Cropping2D, Add, UpSampling2D
import sys
sys.path.insert(1, '../src')
sys.path.insert(1, '../image_segmentation_keras')
from keras_segmentation.models.model_utils import get_segmentation_model
from crfrnn_layer import CrfRnnLayer


def unet_conv_block(inputs, filters, pool=True, batch_norm_first=True):
    if batch_norm_first == True:
        x = Conv2D(filters, 3, padding="same")(inputs)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2D(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
    elif batch_norm_first == False:
        x = Conv2D(filters, 3, padding="same")(inputs)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)

        x = Conv2D(filters, 3, padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)

    if pool == True:
        x = MaxPooling2D((2, 2))(x)
        return x
    else:
        return x

def unet_decoder(encoder,input_height, input_width, n_classes, filters, depth, transpose, batch_norm_first, crfrnn_layer):

    encoder = kwargs[encoder]
    input_height = kwargs[input_height]
    input_width = kwargs[input_width]
    n_classes = kwargs[n_classes]
    filters = kwargs[filters]
    depth = kwargs[depth]
    transpose = kwargs[transpose]
    batch_norm_first = kwargs[batch_norm_first]
    crfrnn_layer = kwargs[crfrnn_layer]

    img_input, levels = encoder
    levels = levels[::-1] #reverses list to start with level with the most filters
    filters = levels[::-1]

    if len(levels) != len(filters):
        msg = "The number of levels and filters need to be the same"
        raise ValueError(msg)

    if transpose == False:
        for i in range(depth):
            x = unet_conv_block(levels[i], pool=False)
            if i < max(range(depth)):
                x = UpSampling2D((2, 2)))(x)
                x = concatenate([x, levels[i+1]], axis=3))
            else:
                x = Conv2D(n_classes, (1, 1), padding='same')(x)
                return x
    elif transpose == True:
        for i in range(depth):
            x = unet_conv_block(levels[i], pool=False)
            if i < max(range(depth)):
                x = Conv2DTranspose(filters[i],(2,2), strides=2, padding='same')(x)
                x = concatenate([x, levels[i+1]], axis=3))
            else:
                x = Conv2D(n_classes, (1, 1) , padding='same')(x)

def one_side_pad(x):
    x = ZeroPadding2D((1, 1))(x)
    x = Lambda(lambda x: x[:, :-1, :-1, :])(x)
    return x

def resnet50_identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at
                     main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters


    bn_axis = 3


    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1),
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1),
               name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def resnet50_conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at
                     main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with
    strides=(2,2) and the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters


    bn_axis = 3

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1),
               name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1),
                      strides=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)
#     print(shortcut.shape)
    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x
