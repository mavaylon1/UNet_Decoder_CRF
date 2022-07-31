from UNet_Decoder_CRF.model_utils import unet_encoder, unet_decoder
from UNet_Decoder_CRF.keras_segmentation import train
from glob import glob
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

input_height = 256
input_width = 256
encoder_filters=[512, 256, 128, 64]
channels = 3
batch_norm_first = False
n_classes = 3
decoder_filters=[1024, 512, 256, 128, 64]

encoder = unet_encoder(input_height=input_height, input_width=input_width, filters=encoder_filters, channels=channels, batch_norm_first=batch_norm_first)
model = unet_decoder(encoder=encoder, input_height=input_height, input_width=input_width, n_classes=n_classes, filters=decoder_filters, model='unet',
                     transpose=False, batch_norm_first=batch_norm_first, crfrnn_layer=True)

model.train(
    train_images =  "/global/homes/m/mavaylon/train/img/",
    train_annotations = "/global/homes/m/mavaylon/train/ann/",
    epochs=20,
    steps_per_epoch=len(glob("/global/homes/m/mavaylon/train/img/*")),
    batch_size=1,
    validate=True,
    val_images="/global/homes/m/mavaylon/test/img/",
    val_annotations="/global/homes/m/mavaylon/test/ann/",
    val_batch_size=1,
    val_steps_per_epoch=len(glob("/global/homes/m/mavaylon/test/img/*"))
)
# model.load_weights('/Users/mavaylon/Research/pet_weights/UNET_CRF_PET/unet87.94pet_class_crf.h5')

# Predictions
# img_names = sorted(glob("/Users/mavaylon/Research/pet_predictions/img/*.png"))
#
# for name in img_names:
#     out_name = "/Users/mavaylon/Research/CNN_CRF/test_pred_repo/unet_CRF_pet/" + name.split('/')[-1]
#     print(out_name)
#     out = model.predict_segmentation(inp=name, out_fname=out_name)
