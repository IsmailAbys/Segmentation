This folder contains python scripts for 3D Unet architechture which is based on integrated MONAI into an existing PyTorch medical DL program. 
The Pelvis dataset can be downloaded here https://zenodo.org/record/4588403#.Ym-WotpBy3B

Dataset: dataset6 (CLINIC) 
Images: CTPelvic1K_dataset6_data.tar.gz
Masks: CTPelvic1K_dataset6_Anonymized_mask.tar.gz
Target: Pelvis
Modality: CT
Format: NIFTI
Size: 60 cases (41 Training + 19 Testing)

Labels: 
0: background, 
1: sacrum, 
2: right_hip, 
3: left_hip, 
4: lumbar_vertebra    

1) train.py file --> 3D Unet training which works with Early Stopping.
2) evaluate.py file --> evaluates the trained model on testing dataset and verify the dice score of each case. 
3) predict.py file --> the inference to upload one image and receive one segmented mask: python predict.py -i image.nii.gz -o seg.nii.gz
4) pytorchtools.py file --> for early stopping and it should be placed in the same folder where train file is place.
5) slicer.py file --> to transform segmented mask and make it without background for 3D slicer visualization: python slicer.py -i seg.nii.gz -o name.nii.gz




