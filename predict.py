"""
Author: Milos Zivkovic
pho.milos@gmail.com
"""

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import numpy as np
import mimetypes
import argparse
import imutils
import cv2
import os

BASE_PATH = "dataset"
MODEL_PATH = "detector.h5"

IMAGES_PATH = os.path.sep.join([BASE_PATH, "images"])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", default="test_images.txt", help="path to input image/text file of image filenames")
    args = vars(ap.parse_args())

    filetype = mimetypes.guess_type(args["input"])[0]
    imagePaths = [args["input"]]
    # if the file type is a text file, then we need to process *multiple* images
    if "text/plain" == filetype:
        # load the filenames in our testing file and initialize our list of image paths
        filenames = open(args["input"]).read().strip().split("\n")
        imagePaths = []
        for f in filenames:
            # get full path name
            imagePaths.append(f)

    # load our trained model
    print("[INFO] loading object detector...")
    model = load_model(MODEL_PATH)
    for imagePath in imagePaths:
        # load the input image (in Keras format) from disk and preprocess it, scale pixel intensities to [0, 1]
        image = load_img(imagePath, target_size=(270, 480))
        image = img_to_array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        # get predictions
        preds = model.predict(image)[0]

        (startX, startY, endX, endY) = preds
        # load the input image, resize it to desired size and get dimensions
        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=600)
        (h, w) = image.shape[:2]
        # scale the predicted bounding box coordinates based on the image dimensions
        startX = int(startX * w)
        startY = int(startY * h)
        endX = int(endX * w)
        endY = int(endY * h)
        # draw the predicted bounding box on the image
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        # show the output image
        cv2.imshow("Output", image)

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
