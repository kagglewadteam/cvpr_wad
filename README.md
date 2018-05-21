# MASK_rcnn_train_custom_data
## Installation
1. clone this repo: 
```
git clone git@github.com:kagglewadteam/cvpr_wad.git --recursive
```
2. see https://github.com/matterport/Mask_RCNN#installation
## Directory Structure
.<br />
├─assets<br />
├─dist<br />
├─images<br />
├─mask_rcnn.egg-info<br />
├─mrcnn<br />
├─mask_rcnn_coco.h5 :heart: <br />
└─samples<br />
&nbsp;   └── .ipynb_checkpoints<br />
&nbsp;    └── balloon<br />
&nbsp;    └── coco<br />
&nbsp;    └── kaggle_wad<br />
&nbsp;    └── nucleus<br />
&nbsp;    └── shapes<br />

the files with heart are newly added. Not neccessary to download the mask_rcnn_coco.h5<br />
it can be downloaded automatically.

## Training
cd into the `kaggle_wad` directory<br />
```bash
cd Mask_RCNN/samples/kaggle_wad
python3 train.py
```
one thing should be noticed that your own directories of images and masks should be defined in the train.py

## Tuning hyper-parameters
 └─mrcnn <br />
&nbsp;   └── config.py :heart: <br />
&nbsp;    └── model.py<br />
&nbsp;    └── parallel_model.py<br />
&nbsp;    └── utils.py <br />
&nbsp;    └── visualize.py <br />
&nbsp;    └── __init__.py<br />

most parameters in the config.py, few in the kaggle_wad/train.py

## Detection
cd into the `kaggle_wad ` directory<br />
```bash
cd Mask_RCNN/samples/kaggle_wad
python3 detect.py
```

## Evaluate Submission
Validate the format of submission csv file.
1. cd into the `submission` directory: 
```bash
cd submission
```
2. Change the directory of test images and submission file accordingly
3. Validate the submission file
```bash
python3 submission_checker.py
```	
