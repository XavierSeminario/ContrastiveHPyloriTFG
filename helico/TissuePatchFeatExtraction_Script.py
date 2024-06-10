# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:14:32 2024

@author: debora
"""
import sys
import os
import numpy as np
import pandas as pd
import glob
import cv2 as cv
from matplotlib import pyplot as plt
from ismember import ismember
from random import shuffle


CodeDir=r'C:\Users\xavis\OneDrive\Escritorio\Uni\TFG\ContrastiveHPyloriTFG\helico' 
sys.path.append(CodeDir)
libDirs=next(os.walk(CodeDir))[1]
for lib in libDirs:
    sys.path.append(os.path.join(CodeDir,lib))



from Extractors import FeatExtractor


    
######################### 0. EXPERIMENT PARAMETERS

# 0.2 FOLDERS
DataDir=r'C:\Users\xavis\OneDrive\Escritorio\Uni\TFG\ContrastiveHPyloriTFG\helico'

os.chdir(DataDir)

ResultsDir=r'C:\Users\xavis\OneDrive\Escritorio\Uni\TFG\ContrastiveHPyloriTFG\helico'
ExcelMetaData=['HP_WSI-CoordAnnotatedWindows.xlsx',
               'HP_WSI-CoordAugAnnotatedWindows.xlsx']
#ExcelMetaData=['HP_WSI-CoordAnnotatedWindows.xlsx']
# 1. LOAD DATA
df_meta=pd.DataFrame()
excelfile=ExcelMetaData[0]
df_meta=pd.concat([df_meta,pd.read_excel(os.path.join(DataDir,excelfile))])
# Data Curation
df_meta=df_meta[(df_meta.Cropped==0)*(df_meta.Deleted!=1)]
df_meta=df_meta.loc[df_meta.Presence!=0]
excelfile=ExcelMetaData[1]
df=pd.read_excel(os.path.join(DataDir,excelfile))
df_meta=pd.concat([df_meta,df])


casesID=[PatID+'_'+str(WSI) for PatID,WSI in 
         zip(df_meta.Pat_ID.values,df_meta.WSI_ID.values)]
winID=df_meta.Window_ID.values
y_true=df_meta.Presence.values==1
y_true=y_true.astype(int)

            
# Load Patches
files=[os.path.join(case,str(win)+'.png') for case,win in zip(casesID,winID)]
patches=[]
winID_patches=[]
casesID_patches=[]
files_patches=[]
y_true_patches=[]
for k in np.arange(len(files)):
    
    file=files[k]
    file=os.path.join(DataDir,'SubImage',file)
    if os.path.isfile(file):
        im = cv.imread(os.path.join(DataDir,'SubImage',file))[:,:,0:3]
        patches.append(im)
        winID_patches.append(winID[k])
        casesID_patches.append(casesID[k])
        files_patches.append(files[k])
        y_true_patches.append(y_true[k])

        
            
    
### 2. PREPROCESSING
patches=np.stack(patches)
y_true_patches=np.array(y_true_patches)
## 2.1 Image normalization
patches=patches.astype('float32')
mu=[np.mean(patches[:,:,:,0].flatten()),
    np.mean(patches[:,:,:,1].flatten()),np.mean(patches[:,:,:,2].flatten())]
std=[np.std(patches[:,:,:,0].flatten()),
     np.std(patches[:,:,:,1].flatten()),np.std(patches[:,:,:,2].flatten())]
for kch in np.arange(3):
    patches[:,:,:,kch]=(patches[:,:,:,kch]-mu[kch])/std[kch]
    
### 3. PRETRAINED FEATURES EXTRACTION

# 3.2 Feature Extraction    

batch_size=100
feRes=FeatExtractor(patches,'ResNet',batch_size)
feDens=FeatExtractor(patches,'DenseNet',batch_size)
feVGG=FeatExtractor(patches,'Vgg16',batch_size)
feEff=FeatExtractor(patches,'EfficientNet',batch_size)

np.savez(os.path.join(DataDir,'PreTrainedFeatures'+'_Immuno_Aug'+'.npz'),feRes=feRes,
         feDens=feDens,feVGG=feVGG,feEff=feEff,y_true=y_true_patches,
         PatID_patches=casesID_patches,winID_patches=winID_patches)

