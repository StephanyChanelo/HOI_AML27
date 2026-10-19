# HAND-OBJECT INTERACTION (HOI) PROJECT ✍️
Hand-Object Interaction (HOI) pose estimation is a fundamental computer vision task aimed at jointly recovering the 3D position, orientation, and joint kinematics of human hands alongside the 6D pose and geometry of manipulated objects from visual input.

# Dataset 
The DexYCB dataset is a large-scale dataset published by NVIDIA and the University of Washington in 2021. It contains around 600K multi-view RGB-D frames of hands interacting with 20 daily objects from the YCB dataset. Including annotations of the segmentation masks, 6D object poses, MANO hand poses and 2D/3D hand joint coordinates.

Dataset page: https://dex-ycb.github.io/

You will need the files:  20200709-subject-01.tar.gz,  bop.tar.gz, calibration.tar.gz, models.tar.gz

Make sure that you have the folders in the correct structure. You can download the zip file from the following link: https://drive.google.com/file/d/1ECo1wFpMtkEtX8i3sFt3FJfr2K6-beuM/view?usp=sharing

# Data Inspection 🔎
```
python dataset_inspection.py --dataset_root "/content/DexYCB" --index 100
```
