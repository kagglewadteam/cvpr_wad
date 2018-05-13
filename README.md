# MASK_rcnn_train_custom_data
## Procedure
- clone and install the mask rcnn
- copy the file kaggle_wad to the samples
- run the train

## Install
git clone https://github.com/matterport/Mask_RCNN.git <br />
and see https://github.com/matterport/Mask_RCNN#installation
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
&nbsp;    └── kaggle_wad :heart: <br />
&nbsp;    └── nucleus<br />
&nbsp;    └── shapes<br />

the files with heart are newly added, kaggle_wad can be copied from my branch. Not neccessary to download the mask_rcnn_coco.h5<br />
it can be downloaded automatically.

## Training
cd into the directory<br />
```
cd kaggle_wad
python3 train.py
```
one thing should be noticed that your own directories of images and masks should be defined in the train.py

# Tuning hyper-parameters
 └─mrcnn <br />
&nbsp;   └── config.py :heart: <br />
&nbsp;    └── model.py<br />
&nbsp;    └── parallel_model.py<br />
&nbsp;    └── utils.py <br />
&nbsp;    └── visualize.py <br />
&nbsp;    └── __init__.py<br />

most parameters in the config.py, few in the kaggle_wad/train.py
