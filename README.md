# HUDStats Ball Detection Solution
Solution for ball detection task

##Important
After cloning the repo use this link to download trained model (It is too big and cannot be commited to git)

https://www.dropbox.com/s/b4ebs2c5ao018c3/detector.h5?dl=0

When downloaded just put it in the repo root dir together with all scripts.
Your ready directory should look like this:
```bash
│   dataset_preprocesing.py
│   detector.h5
│   evaluation.py
│   plot.png
│   predict.py
│   predictions.csv
│   README.md
│   requirements.txt
│   test_images.txt
│   train.py
└───dataset
    │   annotations.csv
    └───images
            12104.png
            14426.png
            14535.png
            19460.png
            2036.png
            20673.png
            22591.png
            22771.png
            22818.png
            24064.png
            3474.png
            3846.png
            4342.png
            443.png
            4814.png
            5195.png
            5464.png
            6729.png
            686.png
            7885.png
```
## Random prediction script
In the folder dataset/images there are 20 random images extracted from the video provided.

Using predict.py script, trained model based on VGG16 architecture will be loaded and predictions will be made and visualised on screen
## Final Evaluation Script
Script evaluation.py does 2 things as it was required.
It is necessary to provide two arguments:
1. -input => path to the video file on which you wish to do predictions
2. -visualise => True/False to visualise results

**In the repo you can find predictions.csv that is already created, these are predictions made on the 'part1.mp4' video. If you wish to predict ball location on another video you will need to delete this .csv file. Then you can launch the script with the new video provided, and after the script completes populating csv file, if you set visualise to ***True*** you can visualise the predicted results for the selected video file

In case you want to visualise predictions from the already provided csv file, provide -input argument (path to the part1.mp4 video) and set -visualise ***True***

**When visualizing predictions user will be prompted to enter number of frame from which he wishes to see consecutive predictions visualised

requirements.txt file is also provided for installing any dependencies one may need to run provided scripts
