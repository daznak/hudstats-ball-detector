"""
Author: Milos Zivkovic
pho.milos@gmail.com
"""

import argparse
import os

import cv2
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import csv
import pandas as pd


showing_width = 960

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="path to input video file of image filenames")
    ap.add_argument("-v", "--visualise", default=False, help="visualise or not")

    args = vars(ap.parse_args())

    # Get arguments
    video_file = args["input"]
    visu_on = args["visualise"]
    # If predictions.csv is not available generate it
    if "predictions.csv" not in os.listdir():
        # Load trained model
        MODEL_PATH = "detector.h5"
        print("[INFO] loading object detector...")
        model = load_model(MODEL_PATH)

        # Load video file
        cap = cv2.VideoCapture(video_file)
        # Get max number of frames
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create predictions.csv file
        file = open('predictions.csv', 'w', newline='')

        with file:
            # Write header
            header = ['Frame_no', 'ball_x1', 'ball_y1', 'ball_x2', 'ball_y2']
            writer = csv.DictWriter(file, fieldnames=header)
            # write data row-wise into the csv file
            writer.writeheader()
            # Read frame
            while cap.isOpened():

                ret, frame = cap.read()
                im = Image.fromarray(frame)
                # Resize image to fit the input dimensions of the trained model
                im = im.resize((480, 270))
                im = img_to_array(im) / 255.0
                image = np.expand_dims(im, axis=0)

                if ret:
                    # Predict bounding box coordinates
                    preds = model.predict(image)[0]
                    print("{}/{}".format(int(cap.get(cv2.CAP_PROP_POS_FRAMES)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
                    (startX, startY, endX, endY) = preds

                    frame = imutils.resize(frame, width=showing_width)
                    (h, w) = frame.shape[:2]
                    # scale the predicted bounding box coordinates based on the image dimensions we want to show
                    startX = int(startX * w)
                    startY = int(startY * h)
                    endX = int(endX * w)
                    endY = int(endY * h)

                    ball_annot = {'Frame_no': int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                                  'ball_x1': startX,
                                  'ball_y1': startY,
                                  'ball_x2': endX,
                                  'ball_y2': endY}
                    # Write predictions into csv
                    writer.writerow(ball_annot)
                else:
                    break

    if "predictions.csv" in os.listdir() and visu_on:
        # Visualisation based on previously created csv file
        print("Predicted CSV file already exists...")
        # Load video
        cap = cv2.VideoCapture(video_file)
        # Get number of frames in video
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Read csv predictions
        df_frames = pd.read_csv(r"predictions.csv")
        df_frames.set_index("Frame_no", inplace=True)

        # User input for frame number from which to show consecutive results
        from_frame = int(input("Insert frame number from which to show consecutive "
                               "results (max frame number = {}):".format(num_frames-1)))

        for frame_no in range(from_frame, num_frames):
            # Go to frame_no in video
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

            # Read frame
            ret, frame = cap.read()
            image = imutils.resize(frame, width=showing_width)
            (h, w) = image.shape[:2]

            # Get prediction
            frame_info = df_frames.loc[frame_no]

            # draw the predicted bounding box on the image
            startX = int(frame_info["ball_x1"])
            startY = int(frame_info["ball_y1"])
            endX = int(frame_info["ball_x2"])
            endY = int(frame_info["ball_y2"])

            cv2.rectangle(image, (startX, startY), (endX, endY), (255, 0, 0), 2)
            # show the output image
            cv2.imshow("Output", image)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
