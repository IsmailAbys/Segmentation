from monai.utils import first, set_determinism
import time
import csv
from pytorchtools import EarlyStopping
from monai.transforms import (
    AddChanneld,
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    Resized,
    CropForegroundd,
    RandScaleIntensityd,
    DataStatsd,
    LoadImaged,
    Orientationd,
    Activationsd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    NormalizeIntensityd,
    RandSpatialCropd,
    ScaleIntensityRanged,
    Spacingd,
    ScaleIntensityd,
    SaveImaged,
    EnsureTyped,
    EnsureType,
    Invertd,
)
from monai.handlers.utils import from_engine
from monai.losses import DiceCELoss
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import matplotlib.pyplot as plt
import monai
import logging
import os
import glob
import sys
import numpy as np
from monai.utils import get_torch_version_tuple, set_determinism

print_config()


train_dir = "C:/Users/Hripsime/OneDrive - ABYS MEDICAL/projects/CTPelvic1K_data/train_dir/"

def main(train_dir):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    set_determinism(seed=0)
    
    
    train_images = sorted(glob.glob(os.path.join(train_dir, "*data.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(train_dir, "*mask_4label.nii.gz")))
    data_dicts = [{"image": image_name, "mask": label_name} for image_name, label_name in zip(train_images, train_labels)]
    train_files, val_files = data_dicts[:-10], data_dicts[-10:]


    # define transforms to augment the dataset
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "mask"]),
            EnsureChannelFirstd(keys=["image", "mask"]),
            Orientationd(keys=["image", "mask"], axcodes="RAS"),
            Spacingd(keys=["image", "mask"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            ScaleIntensityRanged(keys=["image"], a_min=-500, a_max=500,b_min=0.0, b_max=4.0, clip=True,),
            CropForegroundd(keys=["image", "mask"], source_key="image"),
            RandCropByPosNegLabeld(
                keys=["image", "mask"],
                label_key="mask",
                spatial_size=(128, 128, 128),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
        ),
            EnsureTyped(keys=["image", "mask"]),
        ]
    )

    val_transforms = Compose(
    [
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask"]),
        Orientationd(keys=["image", "mask"], axcodes="RAS"),
        Spacingd(keys=["image", "mask"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-500, a_max=500, b_min=0.0, b_max=4.0, clip=True,),
        CropForegroundd(keys=["image", "mask"], source_key="image"),
        
        RandCropByPosNegLabeld(
            keys=["image", "mask"],
            label_key="mask",
            spatial_size=(128, 128, 128),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,),
            
        
        EnsureTyped(keys=["image", "mask"]),
    ]
)


    # create a training data loader
    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=0)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)    #if you are getting a memory error, please change batch_size to 1. 

    # create a validation data loader
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)
    
    # create UNet, DiceCELoss and Adam optimizer  
    device=torch.device("cuda:0")  
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=5,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)  #DiceLoss was also tested, the results were better in case of DiceCELoss
    optimizer = torch.optim.Adam(model.parameters(), 2e-4)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    

    # use amp to accelerate training
    scaler = torch.cuda.amp.GradScaler()
    # enable cuDNN benchmark
    torch.backends.cudnn.benchmark = True    


    max_epochs = 300
    val_interval = 1

    train_values = []
    epoch_loss_values = []
    metric_values = []

    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=5)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=5)])


    #Early stopping with patience (number of epochs to wait if there is no increase of validation mean dice)
    early_stopping = EarlyStopping(patience=50, verbose=True, delta=0.01)  


    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["mask"].to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        epoch_loss += loss.item()
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)


        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = (val_data["image"].to(device),val_data["mask"].to(device),)
                    val_outputs = model(val_inputs)
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)

                metric = dice_metric.aggregate().item()
                dice_metric.reset()
                metric_values.append(metric)
            

            model.eval()
            with torch.no_grad():
                for train_data in train_loader:
                    train_inputs, train_labels = (train_data["image"].to(device),train_data["mask"].to(device),)
                    train_outputs = model(train_inputs) 
                    train_outputs = [post_pred(i) for i in decollate_batch(train_outputs)]
                    train_labels = [post_label(i) for i in decollate_batch(train_labels)]
                    dice_metric(y_pred=train_outputs, y=train_labels)

                train_metric = dice_metric.aggregate().item()
                dice_metric.reset()
                train_values.append(train_metric)
        
            
        epoch_len = len(str(max_epochs))
    
        print_msg = (f'[{epoch:>{epoch_len}}/{max_epochs:>{epoch_len}}] ' +
                     f'Validation Mean Dice: {metric:.5f} ')

        print(print_msg)

        #array = np.array(metric_values)
        #np.savetxt('E:/dicescore.csv', array, delimiter=',')  #You can open this code if you want to save the validation mean dice on csv file.
    
        early_stopping(metric, model)
        
        if early_stopping.early_stop:     #the training should stop on epoch 152 with a mean dice score around 0.95
            print("Early stopping")
            break
        
    torch.save(model.state_dict(), os.path.join(train_dir, "best_metric_model.pth"))    #saving the trained model
    

    # visualize the mean dice when the network trained 
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(metric_values)+1), metric_values, label='Validation Mean Dice')
    plt.plot(range(1,len(train_values)+1), train_values, label='Training Mean Dice')


    # find position of the highest validation mean dice
    maxposs = metric_values.index(max(metric_values))+1 
    plt.axvline(maxposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

    plt.xlabel('Epochs')
    plt.ylabel('Mean Dice')
    plt.ylim(0, 1.0) 
    plt.xlim(0, len(train_values)+1) 
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig('meandice_plot.png', bbox_inches='tight')   


if __name__ == "__main__":
    main(train_dir)     
