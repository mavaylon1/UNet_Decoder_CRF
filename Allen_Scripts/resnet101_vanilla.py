from keras.layers import *
from keras.models import Model
from keras import layers
from keras.layers.merge import concatenate
import sys
sys.path.insert(1, '../image_segmentation_keras')
from keras_segmentation.models.config import IMAGE_ORDERING

from keras_segmentation.models.model_utils import get_segmentation_model
from glob import glob

# sys.path.insert(1, '../Scripts')
# from glob import glob
# from model_utils import *
# from encoders import *
# from decoder import *
def resnet101_encoder(**kwargs):

    input_height = kwargs['input_height']
    input_width = kwargs['input_width']

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
    print(x.shape)
    f2 = one_side_pad(x)
    print(x.shape)

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

def resnet_unet_decoder(**kwargs):

    encoder = kwargs['encoder']
    input_height = kwargs['input_height']
    input_width = kwargs['input_width']
    n_classes = kwargs['n_classes']
    filters = kwargs['filters']
    depth = kwargs['depth']
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
                    # x = CrfRnnLayer(image_dims=(input_height, input_width),
                    #      num_classes=n_classes,
                    #      theta_alpha=160.,
                    #      theta_beta=3.,
                    #      theta_gamma=3.,
                    #      num_iterations=10,
                    #      name='crfrnn')([x, img_input])
                model = get_segmentation_model(img_input, x)
                return model
    elif transpose == True:
        for i in range(depth):
            x = unet_conv_block(levels[i], filters[i], pool=False, batch_norm_first=True)
            if i < max(range(depth)):
                x = Conv2DTranspose(filters[i],(2,2), strides=2, padding='same')(x)
                x = concatenate([x, levels[i+1]], axis=3)
            else:
                x = Conv2DTranspose(filters[i],(2,2), strides=2, padding='same')(x)
                x = unet_output_block(input=x, n_classes=n_classes)
                if crfrnn_layer == True:
                    x = BatchNormalization()(x)
                    # x = CrfRnnLayer(image_dims=(input_height, input_width),
                    #      num_classes=n_classes,
                    #      theta_alpha=160.,
                    #      theta_beta=3.,
                    #      theta_gamma=3.,
                    #      num_iterations=10,
                    #      name='crfrnn')([x, img_input])
                model = get_segmentation_model(img_input, x)
                return model


encoder=resnet101_encoder(input_height=256, input_width=256)
model = resnet_unet_decoder(encoder=encoder, input_height=256, input_width=256, n_classes=3, filters=[2048, 1024, 512, 256, 64], depth=5, transpose=False, batch_norm_first= True, crfrnn_layer=False)

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
