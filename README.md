# Focal Loss for Dense Rotation Object Detection

## Abstract
This repo is based on [Focal Loss for Dense Object Detection](https://arxiv.org/pdf/1708.02002.pdf), and it is completed by [YangXue](https://github.com/yangxue0827).

This is the baseline work of R<sup>3</sup>Det, paper link: [R<sup>3</sup>Det: Refined Single-Stage Detector with Feature Refinement for Rotating Object](https://arxiv.org/abs/1908.05612).

## Performance
More results and trained models are available in the [MODEL_ZOO.md](MODEL_ZOO.md).
### DOTA1.0
| Model |    Backbone    |    Training data    |    Val data    |    mAP   | GPU | Image/GPU | Anchor | Reg. Loss| lr schd | Data Augmentation | Configs |       
|:------------:|:------------:|:------------:|:---------:|:-----------:|:----------:|:-----------:|:---------:|:---------:|:---------:|:---------:|:---------:|     
| RetinaNet (baseline) | ResNet50_v1d 600->800 | DOTA1.0 trainval | DOTA1.0 test | 62.22 | **1X** GeForce RTX 2080 Ti | 1 | H | smooth L1 | 1x | No | cfgs_res50_dota_v4.py |     
| RetinaNet (baseline) | ResNet50_v1d 600->800 | DOTA1.0 trainval | DOTA1.0 test | 62.79 | 8X GeForce RTX 2080 Ti | 1 | H | smooth L1 | **2x** | No | cfgs_res50_dota_v8.py |      
| RetinaNet (baseline) | **ResNet101_v1d** 600->800 | DOTA1.0 trainval | DOTA1.0 test | 64.19 | 1X GeForce RTX 2080 Ti | 1 | H | smooth L1 | 1x | No | cfgs_res101_dota_v9.py |   
| RetinaNet (baseline) | **ResNet152_v1d** 600->800 | DOTA1.0 trainval | DOTA1.0 test | 65.79 | 8X GeForce RTX 2080 Ti | 1 | H | smooth L1 | 2x | No | cfgs_res152_dota_v12.py |
|  |  |  |  |  |  |  |  |  |  |  |  |  |
| RetinaNet (baseline) | ResNet50_v1d 600->800 | DOTA1.0 trainval | DOTA1.0 test | 61.94 | 1X GeForce RTX 2080 Ti | 1 | R | smooth L1 | 1x | No | cfgs_res50_dota_v1.py |
| RetinaNet (baseline) | ResNet50_v1d 600->800 | DOTA1.0 trainval | DOTA1.0 test | 62.25 | **8X** GeForce RTX 2080 Ti | 1 | R | smooth L1 | **2x** | No | cfgs_res50_dota_v10.py |
| RetinaNet | ResNet50_v1d 600->800 | DOTA1.0 trainval | DOTA1.0 test | 68.65 | 1X GeForce RTX 2080 Ti | 1 | R | [**iou-smooth L1**](https://arxiv.org/abs/1811.07126) | 1x | No | cfgs_res50_dota_v5.py |    
|  |  |  |  |  |  |  |  |  |  |  |  |  |
| [R<sup>3</sup>Det](https://github.com/SJTU-Det/R3Det_Tensorflow) | ResNet50_v1d 600->800 | DOTA1.0 trainval | DOTA1.0 test | 65.73 | 8X GeForce RTX 2080 Ti | 1 | H + R | smooth L1 | 2x | No | - |
| [R<sup>3</sup>Det*](https://github.com/SJTU-Det/R3Det_Tensorflow) | ResNet50_v1d 600->800 | DOTA1.0 trainval | DOTA1.0 test | 67.20 | 8X GeForce RTX 2080 Ti | 1 | H + R | smooth L1 | 2x | No | - |
| [R<sup>3</sup>Det](https://github.com/SJTU-Det/R3Det_Tensorflow) | **ResNet101_v1d** 600->800 | DOTA1.0 trainval | DOTA1.0 test | 71.69 | 8X GeForce RTX 2080 Ti | 1 | H + R | smooth L1 | 3x | Yes | - |
| [R<sup>3</sup>Det](https://github.com/SJTU-Det/R3Det_Tensorflow) | **ResNet152_v1d** 600->800 | DOTA1.0 trainval | DOTA1.0 test | 72.81 | 8X GeForce RTX 2080 Ti | 1 | H + R | smooth L1 | **4x** | Yes | - |
| [R<sup>3</sup>Det*](https://github.com/SJTU-Det/R3Det_Tensorflow) | **ResNet152_v1d** 600->800 | DOTA1.0 trainval | DOTA1.0 test | 73.74 | 8X GeForce RTX 2080 Ti | 1 | H + R | smooth L1 | **4x** | Yes | - |
|  |  |  |  |  |  |  |  |  |  |  |  |  |
| **R<sup>3</sup>Det++** | ResNet50_v1d 600->800 | DOTA1.0 trainval | DOTA1.0 test | 68.54 | 8X GeForce RTX 2080 Ti | 1 | H + R | smooth L1 | 2x | No | - |
| R<sup>3</sup>Det++ | **ResNet152_v1d** 600->800 | DOTA1.0 trainval | DOTA1.0 test | 74.41 | 8X GeForce RTX 2080 Ti | 1 | H + R | smooth L1 | 4x | Yes | - |
| R<sup>3</sup>Det++ | ResNet152_v1d **MS** | DOTA1.0 trainval | DOTA1.0 test | 76.56 | 4X GeForce RTX 2080 Ti | 1 | H + R + more | smooth L1 | 6x | Yes | - |

[R<sup>3</sup>Det*](https://github.com/SJTU-Det/R3Det_Tensorflow): R<sup>3</sup>Det with two refinement stages      
**The performance of all models comes from the source [paper](https://arxiv.org/abs/1908.05612).**       

### Visualization
![1](demo1.png)

![2](demo2.png)

## My Development Environment
**docker images: docker pull yangxue2docker/yx-tf-det:tensorflow1.13.1-cuda10-gpu-py3**      
1、python3.5 (anaconda recommend)               
2、cuda 10.0                     
3、[opencv(cv2)](https://pypi.org/project/opencv-python/)       
4、[tfplot 0.2.0](https://github.com/wookayin/tensorflow-plot) (optional)            
5、tensorflow 1.13       
              
## IoU-smooth L1 Loss
**[SCRDet: Towards More Robust Detection for Small, Cluttered and Rotated Objects (ICCV2019)](https://arxiv.org/abs/1811.07126)**    

![1](example.png)

![2](iou_smooth_l1_loss.png)             

## Download Model
### Pretrain weights
1、Please download [resnet50_v1](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz), [resnet101_v1](http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz) pre-trained models on Imagenet, put it to data/pretrained_weights.       
2、**(Recommend in this repo)** Or you can choose to use a better backbone, refer to [gluon2TF](https://github.com/yangJirui/gluon2TF).    
* [Baidu Drive](https://pan.baidu.com/s/1GpqKg0dOaaWmwshvv1qWGg), password: 5ht9.          
* [Google Drive](https://drive.google.com/drive/folders/1BM8ffn1WnsRRb5RcuAcyJAHX8NS2M1Gz?usp=sharing)      

## Compile
```  
cd $PATH_ROOT/libs/box_utils/cython_utils
python setup.py build_ext --inplace (or make)

cd $PATH_ROOT/libs/box_utils/
python setup.py build_ext --inplace
```

## Train

1、If you want to train your own data, please note:  
```     
(1) Modify parameters (such as CLASS_NUM, DATASET_NAME, VERSION, etc.) in $PATH_ROOT/libs/configs/cfgs.py
(2) Add category information in $PATH_ROOT/libs/label_name_dict/lable_dict.py     
(3) Add data_name to $PATH_ROOT/data/io/read_tfrecord.py 
```     

2、Make tfrecord     
For DOTA dataset:      
```  
cd $PATH_ROOT\data\io\DOTA
python data_crop.py
```  

```  
cd $PATH_ROOT/data/io/  
python convert_data_to_tfrecord.py --VOC_dir='/PATH/TO/DOTA/' 
                                   --xml_dir='labeltxt'
                                   --image_dir='images'
                                   --save_name='train' 
                                   --img_format='.png' 
                                   --dataset='DOTA'
```      

3、Multi-gpu train
```  
cd $PATH_ROOT/tools
python multi_gpu_train.py
```

## Eval
```  
cd $PATH_ROOT/tools
python test_dota.py --test_dir='/PATH/TO/IMAGES/'  
                    --gpus=0,1,2,3,4,5,6,7          
``` 

## Tensorboard
```  
cd $PATH_ROOT/output/summary
tensorboard --logdir=.
``` 

![3](images.png)

![4](scalars.png)

## Reference
1、https://github.com/endernewton/tf-faster-rcnn   
2、https://github.com/zengarden/light_head_rcnn   
3、https://github.com/tensorflow/models/tree/master/research/object_detection    
4、https://github.com/fizyr/keras-retinanet     
