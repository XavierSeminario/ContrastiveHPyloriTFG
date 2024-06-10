# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:26:09 2024

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

DataDir=r'C:\Users\xavis\OneDrive\Escritorio\Uni\TFG\ContrastiveHPyloriTFG\helico'
ResDir=r'C:\Users\xavis\OneDrive\Escritorio\Uni\TFG\ContrastiveHPyloriTFG\helico'

# 1. LOAD DATA
data=np.load(os.path.join(DataDir,'PreTrainedFeatures'+'_Immuno_Aug'+'.npz'),
             allow_pickle=True)
feResAug=data['feRes']
feDensAug=data['feDens']
feVGGAug=data['feVGG']
feEffAug=data['feEff']
y_trueAug=data['y_true']

data=np.load(os.path.join(DataDir,'PreTrainedFeatures'+'_Immuno'+'.npz'),
             allow_pickle=True)
feRes=data['feRes']
feDens=data['feDens']
feVGG=data['feVGG']
feEff=data['feEff']
y_true=data['y_true']
         
from sklearn.manifold import TSNE,trustworthiness
network='EfficientNet'
# network='DensNet'
# network='ResNet'
# network='VGG'

embeddings=feEff
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)
Trst=trustworthiness(embeddings, embeddings_2d)

embeddings=feEffAug
tsne = TSNE(n_components=2, random_state=42)
embeddings_2dAug = tsne.fit_transform(embeddings)
TrstAug=trustworthiness(embeddings, embeddings_2dAug)

plt.figure()
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
            cmap='viridis', marker='o')
plt.scatter(embeddings_2d[y_true==1, 0], embeddings_2d[y_true==1, 1], 
            cmap='viridis', marker='o',color='r')
plt.title('Visualització t-SNE dels Embedings '+network)
plt.savefig(os.path.join(ResDir,network))
plt.figure()
plt.scatter(embeddings_2dAug[:, 0], embeddings_2dAug[:, 1], 
            cmap='viridis', marker='o')
plt.scatter(embeddings_2dAug[y_trueAug==1, 0], embeddings_2dAug[y_trueAug==1, 1], 
            cmap='viridis', marker='o',color='r')
plt.title('Visualització t-SNE dels Embedings '+network+'Aug')
plt.savefig(os.path.join(ResDir,network+'Aug'))


# from sklearn.metrics.pairwise import pairwise_distances
# from sklearn.manifold import _utils
# distances = pairwise_distances(embeddings)
# perplexity=30.0
# conditional_P = _utils._binary_search_perplexity(
#     distances, perplexity,False)