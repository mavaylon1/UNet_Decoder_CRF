from UNet_Decoder_CRF.model_utils import unet_encoder, unet_decoder
from UNet_Decoder_CRF.keras_segmentation import train
from glob import glob

input_height = 256
input_width = 256
encoder_filters=[512, 256, 128, 64]
channels = 3
batch_norm_first = False
n_classes = 2
decoder_filters=[1024, 512, 256, 128, 64]

encoder = unet_encoder(input_height=input_height, input_width=input_width, filters=encoder_filters, channels=channels, batch_norm_first=batch_norm_first)
model = unet_decoder(encoder=encoder, input_height=input_height, input_width=input_width, n_classes=n_classes, filters=decoder_filters, model='unet',
                     transpose=False, batch_norm_first=batch_norm_first, crfrnn_layer=True)

model.train(
    train_images = "/home/maavaylon/Binary_Data/BP_lrc_training/img/",
    train_annotations = "/home/maavaylon/Binary_Data/BP_lrc_training/ann/",
    epochs=20,
    steps_per_epoch=len(glob("/home/maavaylon/Binary_Data/BP_lrc_training/img/*")),
    batch_size=1,
    validate=True,
    val_images="/home/maavaylon/Binary_Data/BP_lrc_testing/img/",
    val_annotations="/home/maavaylon/Binary_Data/BP_lrc_testing/ann/",
    val_batch_size=1,
    val_steps_per_epoch=len(glob("/home/maavaylon/Binary_Data/BP_lrc_testing/img/*"))
)
