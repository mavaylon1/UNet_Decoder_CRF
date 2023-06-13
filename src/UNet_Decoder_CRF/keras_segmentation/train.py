import json
import os

from .data_utils.data_loader import image_segmentation_generator, \
    verify_segmentation_dataset
import six
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras import backend as K
from tqdm.keras import TqdmCallback
import tensorflow as tf
import glob
import sys

smooth=1
def dice_coef(y_true, y_pred):

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return ((2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon()))

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def soft_dice_loss(y_true, y_pred, axis=(1, 2, 3)):
    dice_numerator = 2. * K.sum(y_true * y_pred, axis=axis) + 0.00001
    dice_denominator = K.sum(y_true ** 2, axis=axis) + K.sum(y_pred ** 2, axis=axis) + 0.00001
    dice_loss = 1 - K.mean((dice_numerator) / (dice_denominator))
    return dice_loss

def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

def find_latest_checkpoint(checkpoints_path, fail_safe=True):

    # This is legacy code, there should always be a "checkpoint" file in your directory

    def get_epoch_number_from_path(path):
        return path.replace(checkpoints_path, "").strip(".")

    # Get all matching files
    all_checkpoint_files = glob.glob(checkpoints_path + ".*")
    if len(all_checkpoint_files) == 0:
        all_checkpoint_files = glob.glob(checkpoints_path + "*.*")
    all_checkpoint_files = [ff.replace(".index", "") for ff in
                            all_checkpoint_files]  # to make it work for newer versions of keras
    # Filter out entries where the epoc_number part is pure number
    all_checkpoint_files = list(filter(lambda f: get_epoch_number_from_path(f)
                                       .isdigit(), all_checkpoint_files))
    if not len(all_checkpoint_files):
        # The glob list is empty, don't have a checkpoints_path
        if not fail_safe:
            raise ValueError("Checkpoint path {0} invalid"
                             .format(checkpoints_path))
        else:
            return None

    # Find the checkpoint file with the maximum epoch
    latest_epoch_checkpoint = max(all_checkpoint_files,
                                  key=lambda f:
                                  int(get_epoch_number_from_path(f)))

    return latest_epoch_checkpoint

def masked_categorical_crossentropy(gt, pr):
    from keras.losses import categorical_crossentropy
    mask = 1 - gt[:, :, 0]
    return categorical_crossentropy(gt, pr) * mask


class CheckpointsCallback(Callback):
    def __init__(self, checkpoints_path):
        self.checkpoints_path = checkpoints_path

    def on_epoch_end(self, epoch, logs=None):
        if self.checkpoints_path is not None:
            self.model.save_weights(self.checkpoints_path + "." + str(epoch))
            print("saved ", self.checkpoints_path + "." + str(epoch))


def train(model,
          train_images,
          train_annotations,
          input_height=None,
          input_width=None,
          n_classes=None,
          data_size=None,
          verify_dataset=True,
          checkpoints_path=None,
          epochs=5,
          batch_size=1,
          validate=False,
          val_images=None,
          val_annotations=None,
          val_batch_size=1,
          auto_resume_checkpoint=False,
          load_weights=None,
          steps_per_epoch=512,
          val_steps_per_epoch=512,
          gen_use_multiprocessing=False,
          ignore_zero_class=False,
          optimizer_name='adam',
          do_augment=False,
          augmentation_name="aug_fiber",
          callbacks=None,
          custom_augmentation=None,
          other_inputs_paths=None,
          preprocessing=None,
          read_image_type=1  # cv2.IMREAD_COLOR = 1 (rgb),
                             # cv2.IMREAD_GRAYSCALE = 0,
                             # cv2.IMREAD_UNCHANGED = -1 (4 channels like RGBA)
         ):

    n_classes = model.n_classes
    input_height = model.input_height
    input_width = model.input_width
    output_height = model.output_height
    output_width = model.output_width

    if validate:
        assert val_images is not None
        assert val_annotations is not None

    if optimizer_name is not None:

        if ignore_zero_class:
            loss_k = masked_categorical_crossentropy
        else:
            loss_k = 'categorical_crossentropy'

        model.compile(loss=loss_k,
                      optimizer=optimizer_name,
                      metrics=[dice_coef, 'accuracy'])

    if checkpoints_path is not None:
        config_file = checkpoints_path + "_config.json"
        dir_name = os.path.dirname(config_file)

        if ( not os.path.exists(dir_name) )  and len( dir_name ) > 0 :
            os.makedirs(dir_name)

        with open(config_file, "w") as f:
            json.dump({
                "model_class": model.model_name,
                "n_classes": n_classes,
                "input_height": input_height,
                "input_width": input_width,
                "output_height": output_height,
                "output_width": output_width
            }, f)

    if load_weights is not None and len(load_weights) > 0:
        print("Loading weights from ", load_weights)
        model.load_weights(load_weights)

    initial_epoch = 0

    if auto_resume_checkpoint and (checkpoints_path is not None):
        latest_checkpoint = find_latest_checkpoint(checkpoints_path)
        if latest_checkpoint is not None:
            print("Loading the weights from latest checkpoint ",
                  latest_checkpoint)
            model.load_weights(latest_checkpoint)

            initial_epoch = int(latest_checkpoint.split('.')[-1])

    if verify_dataset:
        print("Verifying training dataset")
        verified = verify_segmentation_dataset(train_images,
                                               train_annotations,
                                               n_classes)
        assert verified
        if validate:
            print("Verifying validation dataset")
            verified = verify_segmentation_dataset(val_images,
                                                   val_annotations,
                                                   n_classes)
            # assert verified

    train_gen = image_segmentation_generator(
        train_images, train_annotations,  batch_size,  n_classes,
        input_height, input_width, output_height, output_width,
        do_augment=do_augment, augmentation_name=augmentation_name,
        custom_augmentation=custom_augmentation, other_inputs_paths=other_inputs_paths,
        preprocessing=preprocessing, read_image_type=read_image_type)

    if validate:
        val_gen = image_segmentation_generator(
            val_images, val_annotations,  val_batch_size,
            n_classes, input_height, input_width, output_height, output_width,
            other_inputs_paths=other_inputs_paths,
            preprocessing=preprocessing, read_image_type=read_image_type)

    callbacks = [
       ModelCheckpoint("unet_crf_"+data_size+"_fiber.h5", verbose=1, save_best_only=True, save_weights_only=True, monitor='val_dice_coef'), 
        EarlyStopping(monitor="dice", mode='max', min_delta=.005, patience=5, verbose=1),
        CSVLogger(filename="unet_"+data_size+"_fiber.csv", separator=",", append=True),
        TqdmCallback(verbose=2)
    ]
    print('fit')

    if not validate:
        model.fit(train_gen, steps_per_epoch=steps_per_epoch,
                  epochs=epochs, callbacks=callbacks, initial_epoch=initial_epoch,)
    else:
        model.fit(train_gen,
                  steps_per_epoch=steps_per_epoch,
                  validation_data=val_gen,
                  validation_steps=val_steps_per_epoch,
                  epochs=epochs, callbacks=callbacks)
