"""
Author: Milos Zivkovic
pho.milos@gmail.com
"""

from argparse import ArgumentParser
import numpy as np
import pandas as pd
import cv2
import tqdm
import csv
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split

BASE_PATH = "dataset"
IMAGES_PATH = os.path.sep.join([BASE_PATH, "images"])
ANNOTS_PATH = os.path.sep.join([BASE_PATH, "annotations.csv"])

TEST_FILENAMES = "test_images.txt"


def create_dataset():
    """
    This method creates new csv file that contains labels in the following format:
    ['Filename', 'ball_x1', 'ball_y1', 'ball_x2', 'ball_y2']
    This method also creates and saves images from video to be used for training
    :return:
    """
    video_file = args.video
    labels_file = args.labels

    # Read labels file: should have header
    # [frame_no,ball_x,ball_y] :: frame number, ball x coord in pixels, ball y coord in pixels
    df_frames = pd.read_csv(labels_file)
    df_frames.set_index("frame_no", inplace=True)

    # Load video
    cap = cv2.VideoCapture(video_file)

    # Get number of frames in video
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if 'dataset' not in os.listdir():
        os.mkdir("dataset")
        os.mkdir("dataset/images")
    file = open('dataset/annotations.csv', 'w', newline='')
    with file:

        header = ['Filename', 'ball_x1', 'ball_y1', 'ball_x2', 'ball_y2']
        writer = csv.DictWriter(file, fieldnames=header)
        # writing data row-wise into the csv file
        writer.writeheader()
        # Visualize k random frames
        for frame_no in tqdm.tqdm(range(0, num_frames)):

            # Some frames don't have the ball or it is occluded
            if frame_no not in df_frames.index:
                continue

            # Get frame labels for that frame number
            frame_info = df_frames.loc[frame_no]

            # Go to frame_no in video
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

            # Read frame
            ret, frame = cap.read()

            # Get pos
            ball_pos = np.array([frame_info["ball_x"], frame_info["ball_y"]]).astype(np.int)

            # Drawing points
            pt1, pt2 = tuple(ball_pos - 5), tuple(ball_pos + 5)

            # Resize frame for visualization (original is 1080p_
            cv2.imwrite("dataset/images/{}.png".format(frame_no), frame)

            ball_annot = {'Filename': 'dataset\\images\\{}.png'.format(frame_no),
                          'ball_x1': int(pt1[0]),
                          'ball_y1': int(pt1[1]),
                          'ball_x2': int(pt2[0]),
                          'ball_y2': int(pt2[1])}

            writer.writerow(ball_annot)


def prepare_data():
    print("[INFO] loading dataset...")
    rows = open(ANNOTS_PATH).read().strip().split("\n")

    data = []
    targets = []
    filenames = []

    for row in tqdm.tqdm(rows[1:]):
        row = row.split(",")
        (filename, startX, startY, endX, endY) = row
        image = cv2.imread(filename)
        (h, w) = image.shape[:2]
        # scale the bounding box coordinates relative to the spatial dimensions of the input image
        startX = float(startX) / w
        startY = float(startY) / h
        endX = float(endX) / w
        endY = float(endY) / h

        image = load_img(filename, target_size=(270, 480))
        image = img_to_array(image)
        # update list of data, targets, and filenames
        data.append(image)
        targets.append((startX, startY, endX, endY))
        filenames.append(filename)
    data = np.array(data, dtype="float32") / 255.0
    targets = np.array(targets, dtype="float32")
    # partition the data into training and testing splits
    # 90% of the data for training
    # 10% of the data for testing
    split = train_test_split(data, targets, filenames, test_size=0.10, random_state=42)

    # unpack the data split
    (trainImages, testImages) = split[:2]
    (trainTargets, testTargets) = split[2:4]
    (trainFilenames, testFilenames) = split[4:]
    # write the testing filenames to disk so that we can use then when evaluating/testing our model
    print("[INFO] saving testing filenames...")
    f = open(TEST_FILENAMES, "w")
    f.write("\n".join(testFilenames))
    f.close()
    return trainImages, testImages, trainTargets, testTargets


if __name__ == "__main__":
    """Here we define the argumnets for the script."""
    arg_parser = ArgumentParser()
    arg_parser.add_argument("video", type=str, help="Path to video file")
    arg_parser.add_argument("labels", type=str, help="Path to csv with ball positions")
    args = arg_parser.parse_args()

    create_dataset()
    trainImages, testImages, trainTargets, testTargets = prepare_data()
    # Save created arrays so they can be loaded when we want to train the model
    np.save("trainImages", trainImages)
    np.save("testImages", testImages)
    np.save("trainTargets", trainTargets)
    np.save("testTargets", testTargets)

