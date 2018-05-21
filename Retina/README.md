## 操作流程 <br />
- 安装 keras-retinanet
- 复制训练图片和标签到 keras-retinanet/keras-retinanet/bin 文件夹
- 生成训练用CSV文件
- 进行训练
（如果python2的PIL报错，替换修改过的 [csv_generator.py](https://github.com/kagglewadteam/cvpr_wad/tree/wangmn/csv_generator.py)到keras-retinanet/keras-retinanet/preproccess/csv_generator.py）

## 安装 keras-retinanet
我用的是 <br />
- python2
- tensorflow 1.4
- keras 2.1.5

Clone the repository. https://github.com/fizyr/keras-retinanet<br />
Execute ```pip install . ```<br />
Make sure tensorflow is installed<br />
Make sure Keras 2.1.3 or higher is installed.<br />

## 生成 annotation 和 class map<br />
需要生成两个csv文件， [class_map.csv](https://github.com/kagglewadteam/cvpr_wad/blob/wangmn/class_map.csv) 我已经上传了<br />
还需要生成annotation.csv<br />
生成annotation的函数我已经写好在create_annotation_demo.py<br />
函数需要两个输入,第一个是path_to_img是图片路径,第二个是path_to_label是对应的标签路径， 返回一个list of string格式如下<br />
```
#x1,x2是bounding box的横坐标，y1，y2是纵坐标，x1<x2, y1<y2 
path/to/image.jpg,x1,y1,x2,y2,class_name

#这是为一张图片生成的annotation
train_img/170908_072650121_Camera_5.jpg,2215,1715,2258,1724,car
train_img/170908_072650121_Camera_5.jpg,2771,1723,2782,1736,car
train_img/170908_072650121_Camera_5.jpg,2751,1725,2771,1742,car
train_img/170908_072650121_Camera_5.jpg,2723,1717,2736,1731,car
train_img/170908_072650121_Camera_5.jpg,2749,1721,2760,1733,car
train_img/170908_072650121_Camera_5.jpg,2047,1692,2132,1727,truck
```
你们需要写个代码，循环所有的训练图片的标签，生成完整的annotation.csv<br />

链接是两个CSV文件的具体解释<br />
https://github.com/wangmn93/keras-retinanet/blob/master/README.md#csv-datasets 

## 使用自己的dataset训练<br />
我已经试过这个模型可以跑，下面是我的 keras-retinanet/keras_retinanet/bin 文件夹结构， <br />
训练前需要添加的文件和文件夹我用 :heart: 标出了，<br /> 
两个CSV文件annotation.csv和[class_map.csv](https://github.com/kagglewadteam/cvpr_wad/blob/wangmn/class_map.csv)需要放在文件夹下<br />
需要把训练用的图片放在对应文件夹里，我的文件夹叫train_img，如果要改成别的名字要和annotation.csv里对应<br />
例如我的annotation.csv内容如下<br/>
```
train_img/170908_072650121_Camera_5.jpg,2215,1715,2258,1724,car
train_img/170908_072650121_Camera_5.jpg,2771,1723,2782,1736,car
train_img/170908_072650121_Camera_5.jpg,2751,1725,2771,1742,car
train_img/170908_072650121_Camera_5.jpg,2723,1717,2736,1731,car
train_img/170908_072650121_Camera_5.jpg,2749,1721,2760,1733,car
train_img/170908_072650121_Camera_5.jpg,2047,1692,2132,1727,truck
```

.<br />
├── annotation.csv :heart:<br />
├── class_map.csv :heart:<br />
├── convert_model.py<br />
├── debug.py<br />
├── evaluate_coco.py<br />
├── evaluate.py<br />
├── __init__.py<br />
├── __init__.pyc<br />
├── logs<br />
├── __pycache__<br />
│   └── __init__.cpython-34.pyc<br />
├── snapshots<br />
├── train_img :heart:<br />
│   └── 170908_072650121_Camera_5.jpg :heart:<br />
└── train.py<br />

执行 train.py，如下<br />
我用python2执行的时候,PIL报错，如下
```
  File "../../keras_retinanet/preprocessing/csv_generator.py", line 141, in __init__
    super(CSVGenerator, self).__init__(**kwargs)
  File "../../keras_retinanet/preprocessing/generator.py", line 60, in __init__
    self.group_images()
  File "../../keras_retinanet/preprocessing/generator.py", line 166, in group_images
    order.sort(key=lambda x: self.image_aspect_ratio(x))
  File "../../keras_retinanet/preprocessing/generator.py", line 166, in <lambda>
    order.sort(key=lambda x: self.image_aspect_ratio(x))
  File "../../keras_retinanet/preprocessing/csv_generator.py", line 163, in image_aspect_ratio
    return float(image.width) / float(image.height)
  File "/usr/lib/python2.7/dist-packages/PIL/Image.py", line 528, in __getattr__
    raise AttributeError(name)
AttributeError: width

```
需要替换 keras_retinanet/keras_retinanet/preprocessing/csv_generator.py，修改过的 [csv_generator.py](https://github.com/kagglewadteam/cvpr_wad/tree/wangmn/csv_generator.py) 我已经上传，直接替换<br/>
执行下面这行代码进行训练
```
keras_retinanet/bin/train.py csv /path/to/csv/file/containing/annotations /path/to/csv/file/containing/classes

#这是我执行用的代码
cd keras_retinanet/bin/
python train.py csv annotation.csv class_map.csv

#debug.py可以检查annotation是否正确，它就把图片显示并把object框出来
python debug.py --annotations csv annotation.csv class_map.csv
```