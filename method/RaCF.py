import numpy as np
import tensorflow as tf
from tqdm import tqdm
import pdb
import random
import matplotlib.pyplot as plt
import cv2

def processing_imagenet_image(image,masks,mean= [103.939, 116.779, 123.68]):
    bgr_image = image+mean
    bgr_image=np.where(bgr_image>255,255,bgr_image)
    bgr_image=np.where(bgr_image<0,0,bgr_image)
    rgb_image= bgr_image[0,:,:,::-1]
    mask_image = rgb_image *masks
    mask_image_bgr=mask_image[:,:,:,::-1]
    stack = np.array(mask_image_bgr) - mean
    return stack

class Racf(tf.Module):
    def __init__(self, model,dataset, gpu_batch=300):
        super(Racf, self).__init__()
        self.model = model
        self.dataset = dataset
        self.gpu_batch = gpu_batch

    def load_masks(self, to_path,filepath1,p1):
        self.masks = np.load(to_path+filepath1)
        aa=np.where((np.sum(self.masks,axis=-1))>1,1,np.sum(self.masks,axis=-1))
        mask0 = 1 - aa
        self.mask0 = np.expand_dims(mask0,axis=-1)
        self.N = self.masks.shape[0]
        self.p1 = p1
        return self.masks

    def forward(self, image):
        N = self.N
        image = np.expand_dims(image,axis=0)
        _,  H, W,C = image.shape[0],image.shape[1],image.shape[2],image.shape[3]
        #マスク画像を作成
        if self.dataset == 'GTSRB':
            stack = image *self.masks
        elif self.dataset == 'ImageNet':
            stack = processing_imagenet_image(image,self.masks)
            
        score_list = []
        for i in range(0, N, self.gpu_batch):
            score_list.append(self.model(stack[i:min(i + self.gpu_batch, N)]))
        score_list = np.concatenate(score_list)

        return score_list,self.masks,self.mask0
    
