# Implementation of NeurIPS-2021 paper: Direct Multi-view Multi-person 3D Human Pose Estimation 
# [[paper](https://arxiv.org/pdf/2111.04076.pdf)] [[video-YouTube](https://www.youtube.com/watch?v=dBT4SO2ve0c), [video-Bilibili](https://www.bilibili.com/video/BV1sL4y1v7wy/)] [[slides](https://drive.google.com/file/d/1NJeAYTbbV3ohXaYcUM7up-qITp37pvf6/view)]

This is the official implementation of our NeurIPS-2021 work: Multi-view Pose Transformer (MvP). **MvP is a simple algorithm that directly regresses multi-person 3D human pose from multi-view images.**

**:star::star::star:[News] A Re-implementation is integrated into **xrmocap**: at https://github.com/openxrlab/xrmocap**

## Framework
![mvp_framework](https://github.com/sail-sg/mvp/blob/main/figures/mvp_framework.png)
## Example Result
![mvp_framework](https://github.com/sail-sg/mvp/blob/main/figures/example_qualitative_result.png)

## Reference
```
@article{wang2021mvp,
  title={Direct Multi-view Multi-person 3D Human Pose Estimation},
  author={Tao Wang and Jianfeng Zhang and Yujun Cai and Shuicheng Yan and Jiashi Feng},
  journal={Advances in Neural Information Processing Systems},
  year={2021}
}
```

## 1. Installation
1. Set the project root directory as ${POSE_ROOT}.
2. Install all the required python packages (with requirements.txt).
3. compile deformable operation for projective attention.
```bash
cd ./models/ops
sh ./make.sh
```

## 2. Data and Pre-trained Model Preparation

### 2.1 CMU Panoptic
Please follow [VoxelPose](https://github.com/microsoft/voxelpose-pytorch) to download 
the CMU Panoptic Dataset and PoseResNet-50 pre-trained model.

The directory tree should look like this:
```
${POSE_ROOT}
|-- models
|   |-- pose_resnet50_panoptic.pth.tar
|-- data
|   |-- panoptic
|   |   |-- 16060224_haggling1
|   |   |   |-- hdImgs
|   |   |   |-- hdvideos
|   |   |   |-- hdPose3d_stage1_coco19
|   |   |   |-- calibration_160224_haggling1.json
|   |   |-- 160226_haggling1
|   |   |-- ...
```

### 2.2 Shelf/Campus
Please follow [VoxelPose](https://github.com/microsoft/voxelpose-pytorch) to download 
the Shelf/Campus Dataset. 

Due to the limited and incomplete annotations of the two datasets, we use psudo 
ground truth 3D pose generated from VoxelPose to train the model, we expect mvp would 
perform much better with absolute ground truth pose data. 

Please use voxelpose or other methods to generate psudo ground truth for the training set,
you can also use our generated psudo GT: 
[psudo_gt_shelf](https://drive.google.com/file/d/1eVauegbdLuPHK7KsS3SqkCSsFWvgSeBY/view?usp=sharing). 
[psudo_gt_campus](https://drive.google.com/file/d/1RA3V5RpRiZ3EnA4_HYAUY6Ut7mIW7PMS/view?usp=sharing). 
[psudo_gt_campus_fix_gtmorethanpred](https://drive.google.com/file/d/1doHxMvmInq0aCdN6zqLl59QRJOTWY0zw/view?usp=sharing). 


Due to the small dataset size, we fine-tune Panoptic pre-trained model to Shelf and Campus.
Download the pretrained MvP on Panoptic from 
[model_best_5view](https://drive.google.com/file/d/1kW2KJPvA6t4oOhcLtK_XE63jMGurF1Vb/view?usp=sharing) and 
[model_best_3view_horizontal_view](https://drive.google.com/file/d/1SBEzjWvyObpk1KFgT85JZ9RVSxFySOJN/view?usp=sharing) or 
[model_best_3view_2horizon_1lookdown](https://drive.google.com/file/d/1lrnm6WrshSVqv5HbzLKS7q19Begui6gA/view?usp=sharing)

The directory tree should look like this:
```
${POSE_ROOT}
|-- models
|   |-- model_best_5view.pth.tar
|   |-- model_best_3view_horizontal_view.pth.tar
|   |-- model_best_3view_2horizon_1lookdown.pth.tar
|-- data
|   |-- Shelf
|   |   |-- Camera0
|   |   |-- ...
|   |   |-- Camera4
|   |   |-- actorsGT.mat
|   |   |-- calibration_shelf.json
|   |   |-- pesudo_gt
|   |   |   |-- voxelpose_pesudo_gt_shelf.pickle
|   |-- CampusSeq1
|   |   |-- Camera0
|   |   |-- Camera1
|   |   |-- Camera2
|   |   |-- actorsGT.mat
|   |   |-- calibration_campus.json
|   |   |-- pesudo_gt
|   |   |   |-- voxelpose_pesudo_gt_campus.pickle
|   |   |   |-- voxelpose_pesudo_gt_campus_fix_gtmorethanpred_case.pickle
```

### 2.3 Human3.6M dataset
Please follow [CHUNYUWANG/H36M-Toolbox](https://github.com/CHUNYUWANG/H36M-Toolbox) to prepare the data.


### 2.4 Full Directory Tree

The data and pre-trained model directory tree should look like this, you can only download
the Panoptic dataset and PoseResNet-50 for reproducing the main MvP result and ablation studies:

```
${POSE_ROOT}
|-- models
|   |-- pose_resnet50_panoptic.pth.tar
|   |-- model_best_5view.pth.tar
|   |-- model_best_3view_horizontal_view.pth.tar
|   |-- model_best_3view_2horizon_1lookdown.pth.tar
|-- data
|   |-- pesudo_gt
|   |   |-- voxelpose_pesudo_gt_shelf.pickle
|   |   |-- voxelpose_pesudo_gt_campus.pickle
|   |   |-- voxelpose_pesudo_gt_campus_fix_gtmorethanpred_case.pickle
|   |-- panoptic
|   |   |-- 16060224_haggling1
|   |   |   |-- hdImgs
|   |   |   |-- hdvideos
|   |   |   |-- hdPose3d_stage1_coco19
|   |   |   |-- calibration_160224_haggling1.json
|   |   |-- 160226_haggling1
|   |   |-- ...
|   |-- Shelf
|   |   |-- Camera0
|   |   |-- ...
|   |   |-- Camera4
|   |   |-- actorsGT.mat
|   |   |-- calibration_shelf.json
|   |   |-- pesudo_gt
|   |   |   |-- voxelpose_pesudo_gt_shelf.pickle
|   |-- CampusSeq1
|   |   |-- Camera0
|   |   |-- Camera1
|   |   |-- Camera2
|   |   |-- actorsGT.mat
|   |   |-- calibration_campus.json
|   |   |-- pesudo_gt
|   |   |   |-- voxelpose_pesudo_gt_campus.pickle
|   |   |   |-- voxelpose_pesudo_gt_campus_fix_gtmorethanpred_case.pickle
|   |-- HM36
```



## 3. Training and Evaluation
The evaluation result will be printed after every epoch, the best result can be found in the log.

### 3.1 CMU Panoptic dataset

We train and validate on the five selected camera views. We trained our models on 8 GPUs and batch_size=1 for each GPU, note the total iteration per epoch should be `3205`, if not, please check your data.
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env run/train_3d.py --cfg configs/panoptic/best_model_config.yaml
```

#### Pre-trained models

| Datasets    |  AP<sub>25</sub> |  AP<sub>25</sub> |  AP<sub>25</sub> | AP<sub>25</sub> | MPJPE | pth | 
| :---        |   :---:    |  :---: |  :---:  |  :---:  | :---:  | :---:  |
| Panoptic    |    92.3    |  96.6  |  97.5   | 97.7    | 15.8   | [here](https://drive.google.com/file/d/1kW2KJPvA6t4oOhcLtK_XE63jMGurF1Vb/view?usp=sharing) |


#### 3.1.1 Ablation Experiments

You can find several ablation experiment configs under `./configs/panoptic/`, for example, removing RayConv:

```
python -m torch.distributed.launch --nproc_per_node=8 --use_env run/train_3d.py --cfg configs/panoptic/ablation_remove_rayconv.yaml
```

### 3.2 Shelf/Campus datasets

As shelf/campus are very small dataset with incomplete annotation, we finetune pretrained MvP with pseudo ground truth 3D pose extracted with VoxelPose, we expect more accurate GT would help MvP achieve much higher performance.

```
python -m torch.distributed.launch --nproc_per_node=8 --use_env run/train_3d.py --cfg configs/shelf/mvp_shelf.yaml
```

#### Pre-trained models
| Datasets    |  Actor 1 |  Actor 2 |  Actor 2 | Average | pth | 
| :---        |   :---:  |  :---:   |  :---:   |  :---:  |:---:|
| Shelf       |   99.3   |  95.1    |  97.8    | 97.4 | [here](https://drive.google.com/file/d/1WjM9B4BqRPIkoh-x250kmDIPhqw8GDZ_/view?usp=sharing) |
| Campus      |   98.2   |  94.1    |  97.4    | 96.6 | [here](https://github.com/sail-sg/volo/releases/download/volo_1/d1_384_85.2.pth.tar) |


### 3.3 Human3.6M dataset
MvP also applies to the naive single-person setting, with dataset like Human3.6, to come
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env run/train_3d.py --cfg configs/h36m/mvp_h36m.yaml
```

## 4. Evaluation Only

To evaluate a trained model, pass the config and model pth:

```
python -m torch.distributed.launch --nproc_per_node=8 --use_env run/validate_3d.py --cfg xxx --model_path xxx
```


### LICENSE
This repo is under the Apache-2.0 license. For commercial use, please contact the authors.
