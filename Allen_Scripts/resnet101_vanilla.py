from keras.layers import *
from keras.models import Model
from keras import layers
from keras.layers.merge import concatenate
import sys
sys.path.insert(1, '../image_segmentation_keras')
from keras_segmentation.models.config import IMAGE_ORDERING

from keras_segmentation.models.model_utils import get_segmentation_model
from glob import glob

sys.path.insert(1, '../Scripts')
from glob import glob
from model_utils import *
from encoders import *
from decoder import *


encoder=resnet101_encoder(input_height=256, input_width=256)
model = resnet_unet_decoder(encoder=encoder, input_height=256, input_width=256, n_classes=3, filters=[2048, 1024, 512, 256, 64], transpose=False, batch_norm_first= True, crfrnn_layer=False)

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
