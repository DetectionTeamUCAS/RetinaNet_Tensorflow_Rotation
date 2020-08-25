# Focal Loss for Dense Rotation Object Detection

## Performance（deprecated） 

**Due to the improvement of the code, the performance of this repo is gradually improving, so the experimental results in this file are for reference only.**

### DOTA1.0
| Model |    Backbone    |    Training data    |    Val data    |    mAP   | Model Link | Anchor | Reg. Loss| Angle Range | lr schd | Data Augmentation | GPU | Image/GPU | Configs |      
|:------------:|:------------:|:------------:|:---------:|:-----------:|:----------:|:-----------:|:-----------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|    
| RetinaNet | ResNet50_v1 600->800 | DOTA1.0 trainval | DOTA1.0 test | 53.17 | - | H | smooth L1 | 90 | 1x | No | 8X GeForce RTX 2080 Ti | 1 | cfgs_res50_dota_v3.py |    
| RetinaNet | ResNet50_v1 600->800 | DOTA1.0 trainval | DOTA1.0 test | 63.18 | [model](https://drive.google.com/file/d/18Z3NWhL4gQB5yJLCXBcHBnK-6BPle3m1/view?usp=sharing) | H | smooth L1 | 90 | 1x | No |**1X** GeForce RTX 2080 Ti | 1 | cfgs_res50_dota_v4.py |     
| RetinaNet | ResNet50_v1 600->800 | DOTA1.0 trainval | DOTA1.0 test | 62.79 | - | H | smooth L1 | 90 | **2x** | No | 8X GeForce RTX 2080 Ti | 1 | cfgs_res50_dota_v8.py |     
| RetinaNet | **ResNet101_v1** 600->800 | DOTA1.0 trainval | DOTA1.0 test | 64.73 | - | H | smooth L1 | 90 | 1x | No | 1X GeForce RTX 2080 Ti | 1 | cfgs_res101_dota_v9.py |   
| RetinaNet | **ResNet152_v1** 600->800 | DOTA1.0 trainval | DOTA1.0 test | 66.97 | - | H | smooth L1 | 90 | 1x | No | 2X GeForce RTX 2080 Ti | 1 | cfgs_res152_dota_v12.py |
| RetinaNet | ResNet50_v1 600->800 | DOTA1.0 trainval | DOTA1.0 test | 65.11 | - | H | smooth L1 + **atan(theta)** | 90 | 1x | No | 1X GeForce RTX 2080 Ti | 1 | cfgs_res50_dota_v16.py |     
| RetinaNet | ResNet50_v1 600->800 | DOTA1.0 trainval | DOTA1.0 test | 64.10 | - | H | smooth L1 | **180** | 1x | No | 1X GeForce RTX 2080 Ti | 1 | cfgs_res50_dota_v15.py |     
|  |  |  |  |  |  |  |  |  |  |  |  |  |
| RetinaNet | ResNet50_v1 600->800 | DOTA1.0 trainval | DOTA1.0 test | 62.76 | [model](https://drive.google.com/file/d/1n0O6qLJjdDewb_9FDgsGkISevL7SLD8_/view?usp=sharing) | R | smooth L1 | 90 | 1x | No | 1X GeForce RTX 2080 Ti | 1 | cfgs_res50_dota_v1.py |
| RetinaNet| ResNet50_v1 600->800 | DOTA1.0 trainval | DOTA1.0 test | 62.25 | - | R | smooth L1 | 90 | **2x** | No | **8X** GeForce RTX 2080 Ti | 1 | cfgs_res50_dota_v10.py |
| RetinaNet | ResNet50_v1 600->800 | DOTA1.0 trainval | DOTA1.0 test | 68.65 | - | R | [**iou-smooth L1**](https://arxiv.org/abs/1811.07126) | 90 | 1x | No | 1X GeForce RTX 2080 Ti | 1 | cfgs_res50_dota_v5.py |    

**Some model results are slightly higher than in the paper due to retraining.**