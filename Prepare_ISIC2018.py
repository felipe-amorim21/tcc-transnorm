# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 18:15:43 2019
@author: Reza Azad
"""
import h5py
import numpy as np
import scipy.io as sio
# import scipy.misc as sc
from PIL import Image
import glob

# Parameters
height = 256
width  = 256
channels = 3

############################################################# Prepare ISIC 2018 data set #################################################
Dataset_add = '/ISIC2018/'
Tr_add = 'ISIC2018_Task1-2_Training_Input'

Tr_list = glob.glob(Dataset_add+ Tr_add+'/*.jpg')
# It contains 2594 training samples
Data_train_2018    = np.zeros([2594, height, width, channels])
Label_train_2018   = np.zeros([2594, height, width])

print('Reading ISIC 2018')
for idx in range(len(Tr_list)):
    print(idx+1)
    img = Image.open(Tr_list[idx]).convert('RGB')
    img = img.resize((width, height), Image.Resampling.BILINEAR)
    img = np.array(img, dtype=np.float64)
    Data_train_2018[idx, :,:,:] = img

    b = Tr_list[idx]    
    a = b[0:len(Dataset_add)]
    b = b[len(b)-16: len(b)-4] 
    add = (a+ 'ISIC2018_Task1_Training_GroundTruth/' + b +'_segmentation.png')    
    
    img2 = Image.open(add)
    img2 = img2.resize((width, height), Image.Resampling.BILINEAR)
    img2 = np.array(img2, dtype=np.float64)
    Label_train_2018[idx, :,:] = img2    
         
print('Reading ISIC 2018 finished')

################################################################ Make the train and test sets ########################################    
# We consider 1815 samples for training, 259 samples for validation and 520 samples for testing

Train_img      = Data_train_2018[0:1815,:,:,:]
Validation_img = Data_train_2018[1815:1815+259,:,:,:]
Test_img       = Data_train_2018[1815+259:2594,:,:,:]

Train_mask      = Label_train_2018[0:1815,:,:]
Validation_mask = Label_train_2018[1815:1815+259,:,:]
Test_mask       = Label_train_2018[1815+259:2594,:,:]


np.save('data_train', Train_img)
np.save('data_test' , Test_img)
np.save('data_val'  , Validation_img)

np.save('mask_train', Train_mask)
np.save('mask_test' , Test_mask)
np.save('mask_val'  , Validation_mask)