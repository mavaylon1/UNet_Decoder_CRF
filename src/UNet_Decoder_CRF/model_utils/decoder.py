from keras.layers import Conv2D, MaxPooling2D, Input, ZeroPadding2D, \
    Dropout, Conv2DTranspose, Cropping2D, Add, UpSampling2D, BatchNormalization, Activation
from keras.models import *
from keras.layers.merge import concatenate

from . import unet_conv_block, unet_output_block
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
    model = kwargs['model']
    n_classes = kwargs['n_classes']
    filters = kwargs['filters']
    transpose = kwargs['transpose']
    batch_norm_first = kwargs['batch_norm_first']
    crfrnn_layer = kwargs['crfrnn_layer']

    pool=False

    img_input, levels = encoder
    levels = levels[::-1] #reverses list to start with level with the most filters
    depth = len(filters)
    if len(levels) != depth:
        msg = "The number of levels and filters need to be the same"
        raise ValueError(msg)
    x = levels[0]
    if transpose == False:
        for i in range(depth):
            x = unet_conv_block(x, filters[i], pool, batch_norm_first)
            if i < max(range(depth)):
                x = UpSampling2D((2, 2))(x)
                x = concatenate([x, levels[i+1]], axis=3)
            else:
                if model=='resnet':
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
            x = unet_conv_block(x, filters[i], pool, batch_norm_first)
            print(x.shape)
            if i < max(range(depth)):
                x = Conv2DTranspose(filters[i+1],(2,2), strides=2, padding='same')(x)
                x = concatenate([x, levels[i+1]], axis=3)
            else:
                if model=='resnet':
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
