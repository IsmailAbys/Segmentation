

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
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
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


test_dir  = 'C:/Users/Hripsime/OneDrive - ABYS MEDICAL/projects/CTPelvic1K_data/test_dir'
train_dir = 'C:/Users/Hripsime/OneDrive - ABYS MEDICAL/projects/CTPelvic1K_data/train_dir'

def main(test_dir):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    val_images = sorted(glob.glob(os.path.join(test_dir, "*data.nii.gz")))
    val_labels = sorted(glob.glob(os.path.join(test_dir, "*mask_4label.nii.gz")))
    val_dicts = [ {"image": image_name, "mask": label_name} for image_name, label_name in zip(val_images, val_labels) ]
    val_files = val_dicts[:]


    # define transforms for image and segmentation
    
    val_transforms = Compose([
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        ScaleIntensityRanged(keys=["image"], a_min=-120, a_max=360,b_min=0.0, b_max=1.0, clip=True,),
        CropForegroundd(keys=["image"], source_key="image"),
        EnsureTyped(keys=["image", "mask"]),
    ])
    
   
    # val_ds = Dataset(data=val_files, transform=val_transforms)
    # val_loader = DataLoader(val_ds, batch_size=1, num_workers=3)

    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)
    
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    
    post_transforms = Compose([
        EnsureTyped(keys="pred"),
        Invertd(keys="pred", transform=val_transforms, orig_keys="image", meta_keys="pred_meta_dict", 
            orig_meta_keys="image_meta_dict", meta_key_postfix="meta_dict", nearest_interp=False,to_tensor=True,),
        AsDiscreted(keys="pred", argmax=True, to_onehot=5),
        AsDiscreted(keys="mask", to_onehot=5),
    ])
    
    
    device = torch.device("cuda:0")  
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=5,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)


    model.load_state_dict(torch.load(os.path.join(train_dir, "3DUnet_best_metric_model.pth")))

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
    print (dice_metric.get_buffer()) 
    print("Avarage Metric: ", metric_test)   
    print("Total time seconds: {:.2f}".format((time.time()- start_time)))


if __name__ == "__main__":
    main(test_dir)





