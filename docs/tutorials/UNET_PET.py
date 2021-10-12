from UNet_Decoder_CRF.model_utils import unet_encoder, unet_decoder
from UNet_Decoder_CRF.keras_segmentation import train
from glob import glob

input_height = 256
input_width = 256
encoder_filters=[512, 256, 128, 64]
channels = 3
batch_norm_first = True
n_classes = 3
decoder_filters=[1024, 512, 256, 128, 64]

encoder = unet_encoder(input_height=input_height, input_width=input_width, filters=encoder_filters, channels=channels, batch_norm_first=batch_norm_first)
model = unet_decoder(encoder=encoder, input_height=input_height, input_width=input_width, n_classes=n_classes, filters=decoder_filters, model='unet',
                     transpose=False, batch_norm_first=batch_norm_first, crfrnn_layer=False)

# model.train(
#     train_images =  "/Users/mavaylon/Research/Data1/train/img/",
#     train_annotations = "/Users/mavaylon/Research/Data1/train/ann/",
#     epochs=20,
#     steps_per_epoch=len(glob("/Users/mavaylon/Research/Data1/train/img/*")),
#     batch_size=1,
#     validate=True,
#     val_images="/Users/mavaylon/Research/Data1/test/img/",
#     val_annotations="/Users/mavaylon/Research/Data1/test/ann/",
#     val_batch_size=1,
#     val_steps_per_epoch=len(glob("/Users/mavaylon/Research/Data1/test/img/*"))
# )

model.load_weights('/Users/mavaylon/Research/pet_weights/UNET_VANILLA/86.39/pet_class_crf.h5')

# Predictions
img_names = sorted(glob("/Users/mavaylon/Research/pet_predictions/img/*.png"))

for name in img_names:
    out_name = "/Users/mavaylon/Research/CNN_CRF/test_pred_repo/UNET_PET/" + name.split('/')[-1]
    print(out_name)
    out = model.predict_segmentation(inp=name, out_fname=out_name)
