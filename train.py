"""
Author: Milos Zivkovic
pho.milos@gmail.com
"""

from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


INIT_LR = 1e-4
NUM_EPOCHS = 25
BATCH_SIZE = 16
MODEL_PATH = "detector.h5"
PLOT_PATH = "plot.png"


def train(trainImages, trainTargets, testImages, testTargets, INIT_LR, BATCH_SIZE, NUM_EPOCHS, MODEL_PATH, PLOT_PATH):
    """
    This function initializes the model and trains it
    :param trainImages: train images
    :param trainTargets: train labels
    :param testImages:  test images
    :param testTargets: test labels
    :param INIT_LR: initial learning rate
    :param BATCH_SIZE: batch size
    :param NUM_EPOCHS: number of epochs
    :param MODEL_PATH: path where to save the model
    :param PLOT_PATH: path where to save training and val loss plot
    :return:
    """
    # load the VGG16 network
    # cutting off the head FC layers are left off
    vgg = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(270, 480, 3)))
    # freeze all VGG layers so they will *not* be updated during the training process
    vgg.trainable = False
    # flatten the max-pooling output of VGG
    flatten = vgg.output
    flatten = Flatten()(flatten)
    # construct a fully-connected layer header to output the predicted bounding box coordinates
    bboxHead = Dense(128, activation="relu")(flatten)
    bboxHead = Dense(64, activation="relu")(bboxHead)
    bboxHead = Dense(32, activation="relu")(bboxHead)
    bboxHead = Dense(4, activation="sigmoid")(bboxHead)
    # construct the model
    model = Model(inputs=vgg.input, outputs=bboxHead)

    # initialize the optimizer
    # compile the model
    # summary
    opt = Adam(lr=INIT_LR)
    model.compile(loss="mse", optimizer=opt)
    print(model.summary())
    # train the network for bounding box regression
    print("[INFO] training started...")
    H = model.fit(
        trainImages, trainTargets,
        validation_data=(testImages, testTargets),
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        verbose=1)

    print("[INFO] saving trained model...")
    model.save(MODEL_PATH, save_format="h5")
    # plot the model training history
    N = NUM_EPOCHS
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.title("Bounding Box Regression Loss on Training Set")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(PLOT_PATH)


if __name__ == "__main__":
    # Load arrays for training
    print("[INFO] Loading data...")
    trainImages = np.load("trainImages.npy")
    print("[INFO] training images loaded...")
    testImages = np.load("testImages.npy")
    print("[INFO] test images loaded...")
    trainTargets = np.load("trainTargets.npy")
    print("[INFO] training targets loaded...")
    testTargets = np.load("testTargets.npy")
    print("[INFO] test targets loaded...")

    print("[INFO] Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("[INFO] Using GPU: {}".format(tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)))
    # Call train function to train the model
    train(trainImages, trainTargets, testImages, testTargets, INIT_LR, BATCH_SIZE, NUM_EPOCHS, MODEL_PATH, PLOT_PATH)
