import numpy as np
import tensorflow as tf
from tqdm import tqdm
import pdb
import random
import matplotlib.pyplot as plt
import cv2

#それぞれのカラーのみのマスクを作成
def make_color_mask(N,grid_ch,color):
    masks = np.zeros((N, grid_ch.shape[1],grid_ch.shape[2],grid_ch.shape[3]))

    for n in range(N):
        for i in range(grid_ch.shape[1]):
            for j in range(grid_ch.shape[2]):
                if all(grid_ch[n,i,j] == color):
                    masks[n,i,j] = color

    return masks

def generate_masks(to_path,image_size, N, s, p1, savepath):

    grid = np.random.rand(N, s, s) < p1
    grid = grid.astype('float32')
    grid=np.expand_dims(grid,axis=-1)
    grid0=1-grid
    grid_ch = np.concatenate([grid,grid,grid],axis=-1)
    #color
    
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

    red_mask = make_color_mask(N,grid_ch,[1,0,0])
    green_mask = make_color_mask(N,grid_ch,[0,1,0])
    blue_mask = make_color_mask(N,grid_ch,[0,0,1])
    yellow_mask = make_color_mask(N,grid_ch,[1,1,0])
    purple_mask = make_color_mask(N,grid_ch,[1,0,1])
    light_blue_mask = make_color_mask(N,grid_ch,[0,1,1])
    white_mask = make_color_mask(N,grid_ch,[1,1,1])
    
    masks = np.zeros((N, image_size,image_size,grid_ch.shape[3]))
    red_masks = np.zeros((N, image_size,image_size,grid_ch.shape[3]))
    green_masks = np.zeros((N, image_size,image_size,grid_ch.shape[3]))
    blue_masks = np.zeros((N, image_size,image_size,grid_ch.shape[3]))
    yellow_masks = np.zeros((N, image_size,image_size,grid_ch.shape[3]))
    purple_masks = np.zeros((N, image_size,image_size,grid_ch.shape[3]))
    light_blue_masks = np.zeros((N, image_size,image_size,grid_ch.shape[3]))
    white_masks = np.zeros((N, image_size,image_size,grid_ch.shape[3]))

    for i in tqdm(range(N), desc='Generating filters'):
        
        masks[i, :, :] = cv2.resize(grid_ch[i], (image_size,image_size))
        red_masks[i,:,:] = cv2.resize(red_mask[i], (image_size,image_size))
        green_masks[i, :, :] = cv2.resize(green_mask[i], (image_size,image_size))
        blue_masks[i, :, :] = cv2.resize(blue_mask[i], (image_size,image_size))
        yellow_masks[i, :, :] = cv2.resize(yellow_mask[i], (image_size,image_size))
        purple_masks[i,:,:] = cv2.resize(purple_mask[i], (image_size,image_size))
        light_blue_masks[i, :, :] = cv2.resize(light_blue_mask[i], (image_size,image_size))
        white_masks[i, :, :] = cv2.resize(white_mask[i], (image_size,image_size))

    np.save(to_path+ savepath, masks)
    np.save(to_path+'red_mask.npy', red_masks)
    np.save(to_path+'green_mask.npy', green_masks)
    np.save(to_path+'blue_mask.npy', blue_masks)
    np.save(to_path+'yellow_mask.npy', yellow_masks)
    np.save(to_path+'purple_mask.npy', purple_masks)
    np.save(to_path+'light_blue_mask.npy', light_blue_masks)
    np.save(to_path+'white_mask.npy', white_masks)
