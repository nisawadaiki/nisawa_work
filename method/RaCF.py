import numpy as np
import tensorflow as tf
from skimage.transform import resize
from tqdm import tqdm
import pdb
import random
import matplotlib.pyplot as plt
import cv2

def processing_imagenet_image(image,masks,mean= [103.939, 116.779, 123.68]):
    bgr_image = image+mean
    bgr_image=np.where(bgr_image>255,255,bgr_image)
    bgr_image=np.where(bgr_image<0,0,bgr_image)
    rgb_image= cv2.cvtColor(bgr_image[0].astype(np.uint8), cv2.COLOR_BGR2RGB)
    mask_image = rgb_image *masks
    mask_image_bgr=[cv2.cvtColor(mask_image[i].astype(np.uint8), cv2.COLOR_RGB2BGR) for i in range(mask_image.shape[0])]
    stack = np.array(mask_image_bgr) - mean
    return stack

class Racf(tf.Module):
    def __init__(self, model,dataset, gpu_batch=300):
        super(Racf, self).__init__()
        self.model = model
        self.dataset = dataset
        self.gpu_batch = gpu_batch

    def generate_masks(self,to_path,image_size, N, s, p1, savepath):

        #ランダムにマスクするかしないかを決める
        grid = np.random.rand(N, s, s) < p1
        grid = grid.astype('float32')
        grid=np.expand_dims(grid,axis=-1)
        #チャンネルを１から３次元に
        grid_ch = np.concatenate([grid,grid,grid],axis=-1)
        
        """
        ch_list = [np.array([[1, 1, 1]]),np.array([[1, 0, 0]]), 
                np.array([[0, 1, 0]]),np.array([[0, 0, 1]]),np.array([[1, 1, 0]]),
                np.array([[1, 0, 1]]), np.array([[0, 1, 1]])]
        """
        
        #上記のリストを生成するコード(0番目に[0,0,0]を含むため、ch_numで0番目のリスト選ばないようにする)
        ch_list = [np.array([[(i >> 2) & 1, (i >> 1) & 1, i & 1]]) for i in range(8)]
        ch_num = list(range(1, 8))

        #どのチャンネルを保持するかのマスクを作成
        for n in range(N):
            for i in range(grid_ch.shape[1]):
                for j in range(grid_ch.shape[2]):
                    if grid_ch[n,i,j,0] == 1.0:
                        grid_ch[n,i,j] = ch_list[random.choice(ch_num)]
 

        self.masks = np.zeros((N, image_size,image_size,grid_ch.shape[3]))
        #マスクの大きさを入力と同じに
        for i in tqdm(range(N), desc='Generating filters'):
            self.masks[i, :, :] = cv2.resize(grid_ch[i], (image_size,image_size))

        np.save(to_path+savepath, self.masks)
        aa=np.where((np.sum(self.masks,axis=-1))>1,1,np.sum(self.masks,axis=-1))
        mask0 = 1 - aa
        self.mask0 = np.expand_dims(mask0,axis=-1)
        self.N = N
        self.p1 = p1

        return self.masks

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
    
