from monai.transforms import (
    SaveImaged,    
)
import nibabel
import napari
import numpy as np
import numpy
import nibabel as nib
import os



def main():
    
    image = 'C:/Users/Hripsime/OneDrive - ABYS MEDICAL/Desktop/abys_test/out_seg/Bi/Bi_seg.nii.gz'
    data = nibabel.load(image).get_fdata().astype(int)
    data_multiplied = data * np.arange(data.shape[-1])
    data_labels = data_multiplied.sum(axis=-1)

    path = 'E:/Segmentation'
    converted_array = numpy.array(data_labels, dtype=numpy.int32) 
    nifti_file = nibabel.Nifti1Image(converted_array, None)
    nibabel.save(nifti_file, os.path.join(path, 'Bi.nii.gz'))  


if __name__ == "__main__":
    main()  