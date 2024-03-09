
import numpy as np
import cv2
import pickle
import pdb
from tqdm import tqdm
import matplotlib.pyplot as plt

class Saliency():
    def __init__(self,explainer,test_images,true_labels,N):
        self.explainer = explainer
        self.test_images= test_images
        self.true_labels=true_labels
        self.N = N

    def make_saliency(self,to_path,save_pickle):
        saliency_list=[]
        for i in tqdm(range(self.test_images.shape[0]), desc="Processing"):
            score_list,masks,bias_mask = self.explainer.forward(self.test_images[i])
            score=np.expand_dims(score_list[:,self.true_labels[i]],axis=(-1, -2, -3))
            base_saliency = np.mean(masks*score,axis=0)
            bias_saliency = np.mean(bias_mask*score,axis=0)
            saliency = base_saliency # - bias_saliency
            saliency_list.append(saliency)
        saliency_list=np.array(saliency_list)
        if save_pickle:
            with open(to_path+f"saliency{self.N}.pickle","wb") as aa:
                pickle.dump(saliency_list, aa,protocol=4)
        return saliency_list

    def make_gradcam_saliency(self,to_path,save_pickle):
        saliency_list=[]
        for i in tqdm(range(self.test_images.shape[0]), desc="Processing"):
            score_list,masks,bias_mask = self.explainer.forward(self.test_images[i])
            score=np.expand_dims(score_list[:,self.true_labels[i]],axis=(-1, -2, -3))
            saliency = np.mean(masks*score,axis=0)
            #GradCAMによって評価されない領域を、評価領域の最低値に合わせる
            #たまにsaliencyが0の時があるため、場合わけ
            if saliency.min()==saliency.max():
                saliency_list.append(saliency)
                continue
            aa=np.reshape(saliency,[-1])
            bb=np.sort(aa,axis=-1)
            min_num = np.where(bb>0)[0][0]
            saliency=np.where(saliency<=0,bb[min_num],saliency)
            saliency_list.append(saliency)
        saliency_list=np.array(saliency_list)
        if save_pickle:
            with open(to_path+f"saliency{self.N}.pickle","wb") as aa:
                pickle.dump(saliency_list, aa,protocol=4)
        return saliency_list


    def make_color_saliency(self,to_path,save_pickle):
        red_masks        = np.load(to_path+'red_mask.npy')
        blue_masks       = np.load(to_path+'blue_mask.npy')
        green_masks      = np.load(to_path+'green_mask.npy')
        yellow_masks     = np.load(to_path+'yellow_mask.npy')
        purple_masks     = np.load(to_path+'purple_mask.npy')
        light_blue_masks = np.load(to_path+'light_blue_mask.npy')
        white_masks      = np.load(to_path+'white_mask.npy')

        saliency_list=[]
        cl_saliency_list=[]
        for i in tqdm(range(self.test_images.shape[0]), desc="Processing"):
            score_list,masks,bias_mask = self.explainer.forward(self.test_images[i])
            score=np.expand_dims(score_list[:,self.true_labels[i]],axis=(-1, -2, -3))
            saliency = np.mean(masks*score,axis=0)
            saliency_list.append(saliency)

            #saliency = np.mean(masks*score,axis=0)
            sal_red_masks   =   np.mean(red_masks*score,axis=0)
            sal_blue_masks    =  np.mean(blue_masks*score,axis=0)
            sal_green_masks    = np.mean(green_masks*score,axis=0)
            sal_yellow_masks    =np.mean(yellow_masks*score,axis=0)
            sal_purple_masks    =np.mean(purple_masks*score,axis=0)
            sal_light_blue_masks=np.mean(light_blue_masks*score,axis=0)
            sal_white_masks     =np.mean(white_masks*score,axis=0)

            cl_saliency_list.append([sal_red_masks,sal_green_masks,sal_blue_masks,sal_yellow_masks,sal_purple_masks,sal_light_blue_masks,sal_white_masks ])
        #saliency_list=np.array(saliency_list)
        cl_saliency_list=np.array(cl_saliency_list)
        
        #with open(to_path+f"saliency{N}.pickle","wb") as aa:
        #    pickle.dump(saliency_list, aa,protocol=4)
        if save_pickle:
            with open(to_path+f"color_saliency{N}.pickle","wb") as aa:
                pickle.dump(cl_saliency_list, aa,protocol=4)
        
        return cl_saliency_list