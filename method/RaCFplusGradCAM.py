import numpy as np
import tensorflow as tf
from skimage.transform import resize
from tqdm import tqdm
import pdb
import random
import matplotlib.pyplot as plt
import cv2

def processing_imagenet_image(image,masks,gradcam,mean= [103.939, 116.779, 123.68]):
    mean = [103.939, 116.779, 123.68]
    sample = x+mean
    sample=np.where(sample>255,255,sample)
    sample=np.where(sample<0,0,sample)
    sample1 = cv2.cvtColor(sample[0].astype(np.uint8), cv2.COLOR_BGR2RGB)
    grad_mask = masks*gradcam
    mask_image = sample1 *grad_mask
    mask_image_bgr=[cv2.cvtColor(mask_image[i].astype(np.uint8), cv2.COLOR_RGB2BGR) for i in range(mask_image.shape[0])]
    stack = np.array(mask_image_bgr) - mean
    return stack

class RaCF_GradCAM(tf.Module):
    def __init__(self, model, dataset,layer_name, gpu_batch=300):
        super(RaCF_GradCAM, self).__init__()
        self.model = model
        self.dataset = dataset
        self.layer_name = layer_name
        self.gpu_batch = gpu_batch

    def grad_cam(self,x):
        #画像の前処理
        preprocessed_input = np.expand_dims(x, axis=0)
        grad_model_last = tf.keras.models.Model([self.model.inputs], [self.model.get_layer(self.layer_name).output, self.model.output])

        with tf.GradientTape(persistent=True) as tape1:
            tensor_input=tf.convert_to_tensor(preprocessed_input)
            conv_outputs,predict = grad_model_last(tensor_input)    
            label = tf.argmax(predict[0])
            score = predict[:,label]        
            tape1.watch(conv_outputs)

        # 勾配を計算
        grads = tape1.gradient(score, conv_outputs)[0] 
        gate_r = tf.cast(grads > 0, 'float32')
        output = conv_outputs[0]
        gate_f = tf.cast(output > 0, 'float32')

        guided_grads = grads*gate_f*gate_r
        
        # 重みを平均化して、レイヤーの出力に乗じる
        weights = np.mean(guided_grads, axis=(0, 1))
        cam = np.dot(output,weights)
        original_size=(x.shape[1],x.shape[0])
        cam = cv2.resize(cam, original_size ,cv2.INTER_LINEAR)

        cam  = np.maximum(cam, 0)
        # ヒートマップを計算
        if cam.max()==cam.min():
            return cam
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

    def load_masks(self, to_path,filepath1,p1):
        self.masks = np.load(to_path+filepath1)
        self.N = self.masks.shape[0]
        self.p1 = p1
        return self.masks

    def forward(self, x):
        gradcam = self.grad_cam(x)
        #閾値より大きい箇所だけにサンプリング対象を限定
        sort_sal=np.sort(np.reshape(gradcam,[-1]))
        threshold = sort_sal[int(len(sort_sal)*0.7)]
        gradcam=np.where(gradcam>threshold,1,0)
        x = np.expand_dims(x,axis=0)
        gradcam = np.expand_dims(gradcam,axis=(0,-1))
        _,  H, W,C = x.shape[0],x.shape[1],x.shape[2],x.shape[3]

        if self.dataset == 'GTSRB':
            grad_mask = self.masks*gradcam
            stack = grad_mask * x
        elif self.dataset == 'ImageNet':
            stack = processing_imagenet_image(image,self.masks,gradcam)

        score_list = []
        for i in range(0, self.N, self.gpu_batch):
            score_list.append(self.model(stack[i:min(i + self.gpu_batch, self.N)]))
        score_list = np.concatenate(score_list)

        return score_list,grad_mask,_
    
