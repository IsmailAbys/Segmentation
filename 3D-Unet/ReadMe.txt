This folder contains python scripts for 3D Unet architechture which is based on integrated MONAI into an existing PyTorch medical DL program. 
The Pelvis dataset can be downloaded from https://github.com/MIRACLE-Center/CTPelvic1K

Dataset: dataset6 (CLINIC) 
Target: Pelvis
Modality: CT
Format: NIFITI
Size: 60 cases (41 Training + 19 Testing)

Labels: 
0: background, 
1: sacrum, 
2: right_hip, 
3: left_hip, 
4: lumbar_vertebra    

1) training.py file is for 3D Unet training.

2) evaluate.py file is to evaluate the trained model on testing dataset and verify the dice score of each case. 

3) predict.py file is do an inference by uploading one image and receive one segmented mask. The short command is possible for the inference mode by chosing one image from the corresponding folder and segmented mask will be saved in the same folder. 

python predict.py -i image.nii.gz -o seg.nii.gz




