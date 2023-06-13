from UNet_Decoder_CRF.model_utils import unet_encoder, unet_decoder
from UNet_Decoder_CRF.keras_segmentation import train
from glob import glob

input_height = 288
input_width = 288
encoder_filters=[512, 256, 128, 64]
channels = 3
batch_norm_first = False
n_classes = 2
decoder_filters=[1024, 512, 256, 128, 64]

encoder = unet_encoder(input_height=input_height, input_width=input_width, filters=encoder_filters, channels=channels, batch_norm_first=batch_norm_first)
model = unet_decoder(encoder=encoder, input_height=input_height, input_width=input_width, n_classes=n_classes, filters=decoder_filters, model='unet',
                     transpose=False, batch_norm_first=batch_norm_first, crfrnn_layer=False)

# model.load_weights('/global/homes/m/mavaylon/PM_UNET/UNet_Decoder_CRF/docs/tutorials/unet_non_crf__quarter_fiber_2_run.h5')

model.train(
    train_images = "/pscratch/sd/m/mavaylon/Fiber/Quarter/train/img/",
    train_annotations = "/pscratch/sd/m/mavaylon/Fiber/Quarter/train/ann/",
    data_size='non_crf_quarter',
    epochs=10,
    steps_per_epoch=len(glob("/pscratch/sd/m/mavaylon/Fiber/Quarter/train/img/*")),
    batch_size=1,
    verify_dataset=False,
    do_augment=True,
    validate=True,
    val_images="/pscratch/sd/m/mavaylon/Fiber/Quarter/val/img/",
    val_annotations="/pscratch/sd/m/mavaylon/Fiber/Quarter/val/ann/",
    val_batch_size=1,
    val_steps_per_epoch=len(glob("/pscratch/sd/m/mavaylon/Fiber/Quarter/val/img/*"))
)
