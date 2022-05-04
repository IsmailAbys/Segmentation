# Segmentation

This folder contains python scripts for 3D Unet architechture which is based on integrated MONAI into an existing PyTorch medical DL program. 
The Pelvis dataset can be downloaded here https://zenodo.org/record/4588403#.Ym-WotpBy3B
The cases that should be used for training and testing are mentioned in "Dataset.xlsx" file. 
Create "train_dir" and "test_dir" folders where will be placed corresponding datasets (images with their masks).

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

1) train.py --> 3D Unet training which works with Early Stopping.
   In this file you need to modify only "train_dir" and write the path where your training dataset is stored. 
   If you are getting a memory error (CUDA out of memory), in "train_loader" change batch_size to 1.

2) evaluate.py --> evaluates the trained model on testing dataset and verify the dice score of each case. 
   In this file you need to modify only "train_dir" and "test_dir". You need to write paths where your training and testing datasets are stored. 

3) predict.py --> the inference to upload one image and receive one segmented mask: python predict.py -i image.nii.gz -o seg.nii.gz
   In this file you need to modify only "model_dir" and write the path where the model file "best_metric_model.pth" is saved after training.
   
4) slicer.py --> to transform segmented mask and make it without background for 3D slicer visualization: python slicer.py -i seg.nii.gz -o name.nii.gz
   In this file you don't need to modify anything, just place the file in the same folder where is a segmented mask to run the script.

5) pytorchtools.py --> for early stopping and it should be placed in the same folder where the train file is located. README files describe the utility of the code and Requirements files include information about libraries used.
