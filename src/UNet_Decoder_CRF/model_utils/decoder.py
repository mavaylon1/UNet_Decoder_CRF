from keras.layers import Conv2D, MaxPooling2D, Input, ZeroPadding2D, \
    Dropout, Conv2DTranspose, Cropping2D, Add, UpSampling2D, BatchNormalization, Activation
from keras.models import *
from . import unet_conv_block
from ..keras_segmentation.models.model_utils import get_segmentation_model
from ..crf.crfrnn_layer import CrfRnnLayer

"""
    All decoders so far are based on UNET. There are different UNET decorders
    depending on the type of encoder. This was done due to the dimensionality
    constraints from using certain encoder architectures.
"""

def unet_decoder(**kwargs):

    encoder = kwargs['encoder']
    input_height = kwargs['input_height']
    input_width = kwargs['input_width']
    n_classes = kwargs['n_classes']
    filters = kwargs['filters']
    depth = len(filters)
    transpose = kwargs['transpose']
    batch_norm_first = kwargs['batch_norm_first']
    crfrnn_layer = kwargs['crfrnn_layer']

    img_input, levels = encoder
    levels = levels[::-1] #reverses list to start with level with the most filters
    # filters = filters[::-1]

    if len(levels) != len(filters):
        msg = "The number of levels and filters need to be the same"
        raise ValueError(msg)
    x = levels[0]
    if transpose == False:
        for i in range(depth):
            x = unet_conv_block(x, filters[i], pool=False, batch_norm_first=True)
            if i < max(range(depth)):
                x = UpSampling2D((2, 2))(x)
                x = concatenate([x, levels[i+1]], axis=3)
            else:
                x = unet_output_block(input=x, n_classes=n_classes)
                if crfrnn_layer == True:
                    x = BatchNormalization()(x)
                    x = CrfRnnLayer(image_dims=(input_height, input_width),
                         num_classes=n_classes,
                         theta_alpha=160.,
                         theta_beta=3.,
                         theta_gamma=3.,
                         num_iterations=10,
                         name='crfrnn')([x, img_input])
                model = get_segmentation_model(img_input, x)
                return model
    elif transpose == True:
        for i in range(depth):
            x = unet_conv_block(x, filters[i], pool=False, batch_norm_first=True)
            print(x.shape)
            if i < max(range(depth)):
                x = Conv2DTranspose(filters[i+1],(2,2), strides=2, padding='same')(x)
                x = concatenate([x, levels[i+1]], axis=3)
            else:
                out = unet_output_block(input=x, n_classes=n_classes)
                print(out.shape)
                if crfrnn_layer == True:
                    x = BatchNormalization()(x)
                    x = CrfRnnLayer(image_dims=(input_height, input_width),
                         num_classes=n_classes,
                         theta_alpha=160.,
                         theta_beta=3.,
                         theta_gamma=3.,
                         num_iterations=10,
                         name='crfrnn')([x, img_input])
                model = get_segmentation_model(img_input, out)
                return model

def resnet_unet_decoder(**kwargs):

    encoder = kwargs['encoder']
    input_height = kwargs['input_height']
    input_width = kwargs['input_width']
    n_classes = kwargs['n_classes']
    filters = kwargs['filters']
    depth = len(filters)
    transpose = kwargs['transpose']
    batch_norm_first = kwargs['batch_norm_first']
    crfrnn_layer = kwargs['crfrnn_layer']

    img_input, levels = encoder
    levels = levels[::-1] #reverses list to start with level with the most filters
    # filters = filters[::-1]

    if len(levels) != len(filters):
        msg = "The number of levels and filters need to be the same"
        raise ValueError(msg)
    x = levels[0]
    if transpose == False:
        for i in range(depth):
            x = unet_conv_block(x, filters[i], pool=False, batch_norm_first=True)
            if i < max(range(depth)):
                x = UpSampling2D((2, 2))(x)
                x = concatenate([x, levels[i+1]], axis=3)
            else:
                x = UpSampling2D((2, 2))(x)
                x = unet_output_block(input=x, n_classes=n_classes)
                if crfrnn_layer == True:
                    x = BatchNormalization()(x)
                    x = CrfRnnLayer(image_dims=(input_height, input_width),
                         num_classes=n_classes,
                         theta_alpha=160.,
                         theta_beta=3.,
                         theta_gamma=3.,
                         num_iterations=10,
                         name='crfrnn')([x, img_input])
                model = get_segmentation_model(img_input, x)
                return model
    elif transpose == True:
        for i in range(depth):
            x = unet_conv_block(x, filters[i], pool=False, batch_norm_first=True)
            if i < max(range(depth)):
                x = Conv2DTranspose(filters[i+1],(2,2), strides=2, padding='same')(x)
                x = concatenate([x, levels[i+1]], axis=3)
            else:
                x = Conv2DTranspose(filters[i],(2,2), strides=2, padding='same')(x)
                x = unet_output_block(input=x, n_classes=n_classes)
                if crfrnn_layer == True:
                    x = BatchNormalization()(x)
                    x = CrfRnnLayer(image_dims=(input_height, input_width),
                         num_classes=n_classes,
                         theta_alpha=160.,
                         theta_beta=3.,
                         theta_gamma=3.,
                         num_iterations=10,
                         name='crfrnn')([x, img_input])
                model = get_segmentation_model(img_input, x)
                return model
