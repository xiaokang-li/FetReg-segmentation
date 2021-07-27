# FetReg-segmentation
Placental Vessel Segmentation and Registration in Fetoscopy

### Group members

- Chiara Di Vece, University College London
- Juan Pablo de la Cruz Gutiérrez, Infineon Technologies AG
- Mengya Xu, National University of Singapore
- Tiantian Li, University of California, Davis
- Xiaokang Li, Fudan University

### Supervisors
- Sophia Bano & Francisco Vasconcelos
- University College London

## Introduction

**This repository is an implementation of Task 1.**

The project involves solving the following two tasks:
- Task 1: Placental semantic segmentation
- Task 2: Placental RGB frame registration for mosaicking

Task 1:
Placental semantic segmentation
1. Implement Segmentation Architecture (other than Unet ) for placental vessel segmentation
2. Train the architecture for Fold 1 only from [Bano et al. MICCAI2020] ( because of time limitation)
3. Compare the objected results with the baseline results for fold 1 as reported in [Bano et al. MICCAI2020]

Task 2:
Placental RGB frame registration for mosaicking
1. Implement a direct pairwise consecutive frames registration method using either RGB or vessel maps (or
both) provided in the [Bano et al. MICCAI2020] sequence data
2. Use the provided Registration visualisation code to generate the mosaic for at least video 1 sequence

Github
repo: https://github.com/sophiabano/Placental-Vessel-Segmentation-Registration

Contains:
- Segmentation code ( UNet ) from [Bano et al. MICCAI2020]
- Mosaicking visualisation code with sample example

## Dataset
Placental Vessel Dataset: www.ucl.ac.uk/interventional-surgical-sciences/fetoscopy-placenta-data

Segmentation dataset: video01, video02, video03, video04, video05, video06

Split it into valid and test subsets:

| subset | videos |
| :--- | :--- |
| train | 02, 03, 04, 05 |
| vaid | 06 | 
| test | 01 |

For Cross Validation, you can split the dataset like this:

| subset | videos |
| :--- | :--- |
| train | 02, 03, 04, 05, 06 |
| vaid | 01 | 

## Args in main_linux.py

| parm | type | range |
| :--- | :--- | :----- |
| mode | str | train, test, predict |
| model_type | str | Unet, UnetPLusPlus, DeepLabV3 |
| encoder |str | vgg16, resnet50, resnet101, et al. |
| loss_type |str | dice, jaccard, bce, bce_logits |
| lr | float | 1e-3, 3e-4, 1e-4 et al. |
| num_epochs | int | 40 et al. |
| batch_szie | int | 8, 12, 16 et al. |
| model_path | str | 'models/' |
| test_image_index | int | 0, 1, 2 et al.|
| GPU_index | str | “0”, “0, 1, 2”, et al.|

## Pretrained models

Pretrained models can be downloaded here:  
Baidu: https://pan.baidu.com/s/1p83y3MQiDnmgzoPIHsm4Yg password：5yym  
Google Drive:

## How to run this repository

For Windows:
1. Download the dataset and split it into train, valid and test subsets. Put them in the folder 'data/'.
2. Run main.py

For Linux or MAC:
1. Download the dataset and split it into train, valid and test subsets. Put them in the folder 'data/'.
2. Train
   ```shell script
   python main_linux.py --mode train --model_type Unet --encoder vgg16 --loss_type dice --lr 1e-4 --num_epochs 40 --batch_size 16 --model_path models/ --GPU_index 0
   ```
3. Test
   ```shell script
   python main_linux.py --mode test --model_type Unet --encoder vgg16 --loss_type dice --lr 1e-4 --model_path models/ --GPU_index 0
   ```
4. Predict single image in test dir
   ```shell script
   python main_linux.py --mode predict --model_type Unet --encoder vgg16 --loss_type dice --lr 1e-4 --model_path models/ --GPU_index 0 --test_image_index 0
   ```

## Reference
- Segmentation models pytorch: https://github.com/qubvel/segmentation_models.pytorch
- Bano, S., Vasconcelos, F., Shepherd, L.M., Vander Poorten , E., Vercauteren , T., Ourselin , S., David, A.L., Deprest , J. and Stoyanov,
D., 2020, October. Deep placental vessel segmentation for fetoscopic mosaicking . In International Conference on Medical Image
Computing and Computer Assisted Intervention (pp. 763 773). Springer, Cham.
- Bano, S., Casella, A., Vasconcelos, F., Moccia, S., Attilakos , G., Wimalasundera , R., David, A.L., Paladini , D., Deprest , J., Mattos,
L.S. and Stoyanov, D., 2021. FetReg: Placental Vessel Segmentation and Registration in Fetoscopy Challenge Dataset . arXiv
preprint arXiv:2106.05923.
- Bano, Sophia, Francisco Vasconcelos, Marcel Tella Amo, George Dwyer, Caspar Gruijthuijsen , Emmanuel Vander Poorten , Tom
Vercauteren , Sebastien Ourselin , Jan Deprest , and Danail Stoyanov. Deep learning based fetoscopic mosaicking for field of view
expansion. International journal of computer assisted radiology and surgery 15, no. 11 (2020): 1807 1816.
Szeliski , R., 2006. Image alignment and stitching: A tutorial . Foundations and Trends® in Computer Graphics and Vision, 2(1), pp.1-104.
- Placental Vessel Dataset : www.ucl.ac.uk/interventional-surgical-sciences/fetoscopy-placenta-data
- MICCAI2021 EndoVis challenges: https://endovis.grand-challenge.org/
- MATLAB intensity based image registration: https://uk.mathworks.com/help/images/ref/imregister.html
- Image Registration using Enhanced Correlation Coefficient (ECC) Maximization: https://learnopencv.com/image-alignment-ecc-in-opencv-c-python/
- Pystackreg https://pypi.org/project/pystackreg/
- SIMPLEITK: https://simpleitk.readthedocs.io/en/master/registrationOverview.html

