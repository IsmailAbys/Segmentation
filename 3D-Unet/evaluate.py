import logging
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import glob
import time
import monai
from monai.handlers.utils import from_engine
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from torch.utils.data import Dataset
import pandas as pd
from monai.networks.nets import UNet
from monai.transforms import (
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    Invertd,
)

root_dir = "D:/Data"
csv_path = "D:/Data/data_github.csv"
image_folder = "D:/Data/CTPelvic1K_dataset6_data"
mask_folder = "D:/Data/ipcai2021_dataset6_Anonymized"


def main(root_dir):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    #Data Loading using csv file
    class CustomDataset(Dataset):
        def __init__(self, csv_path, image_folder, mask_folder):
            self.data = pd.read_csv(csv_path)
            self.image_folder=image_folder
            self.mask_folder=mask_folder
            return
        
        def __len__(self):
            return len(self.data)
            
        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()
            image = self.image_folder + "/" + self.data['test_images'][idx]
            label = self.mask_folder + "/" + self.data['test_masks'][idx]
            sample = [{"image": image, "mask": label} for image, label in zip(image, label)]
            return sample


    data_dicts = CustomDataset(csv_path, image_folder, mask_folder)
    val_files = data_dicts[:]


    # define pre transforms
    
    val_transforms = Compose([
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        ScaleIntensityRanged(keys=["image"], a_min=-120, a_max=360,b_min=0.0, b_max=1.0, clip=True,),
        CropForegroundd(keys=["image"], source_key="image"),
        EnsureTyped(keys=["image", "mask"]),
    ])
    
   
    #Data Loading and augmentation
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)
    
    #Define a DiceMetric
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    
    # define post transforms
    post_transforms = Compose([
        EnsureTyped(keys="pred"),
        Invertd(keys="pred", transform=val_transforms, orig_keys="image", meta_keys="pred_meta_dict", 
            orig_meta_keys="image_meta_dict", meta_key_postfix="meta_dict", nearest_interp=False,to_tensor=True,),
        AsDiscreted(keys="pred", argmax=True, to_onehot=5),
        AsDiscreted(keys="mask", to_onehot=5),
    ])
    
    
    #create 3D UNet architecture
    device = torch.device("cuda:0")   #define the device which should be either GPU or CPU
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=5,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)


    #loading the saved model of previously trained 3D U-Net model
    model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))

    
    #evaluation of the model by checking the dice score for testing cases
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for val_data in val_loader:
            val_inputs = val_data["image"].to(device)
            roi_size = (160, 160, 160)
            sw_batch_size = 4

            val_data["pred"] = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model,sw_device="cuda:0", device="cpu")
       
            val_data = [post_transforms(i) for i in decollate_batch(val_data)]
            val_outputs, val_labels = from_engine(["pred", "mask"])(val_data)    
        
            
            # compute metric for current iteration
            dice_metric(y_pred=val_outputs, y=val_labels)

        # aggregate the final mean dice result
        metric_test = dice_metric.aggregate().item()
        # reset the status for next validation round
        # dice_metric.reset()    
            
        
    #print("Metric on original image spacing: ", metric_test)
    print (dice_metric.get_buffer())                              #"get_buffer" helps to see dice score of each labels for each case
    print("Avarage Metric: ", metric_test)   
    print("Total time seconds: {:.2f}".format((time.time()- start_time)))


if __name__ == "__main__":
    main(root_dir)





