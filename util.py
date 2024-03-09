import argparse
import matplotlib.pylab as plt
import numpy as np
import os
import pdb

def parser_run():
    parser = argparse.ArgumentParser()
    parser.add_argument("-device_num",type=int, default=0,help='GPU device:deep1 [0] or [1]')
    parser.add_argument("-data",type=str, default="GTSRB",help= 'dataset use [GTSRB] or [ImageNet]')
    parser.add_argument("--train",action="store_true",help='GTSRB')
    parser.add_argument("-mode",type=str, default="RaCF",help= 'CAM_mode: [RaCF] , [RaCF_GradCAM] , [MC-RISE] , [eval]')
    parser.add_argument("--make_mask",action="store_true",help='True:make_mask, False:load_mask')
    parser.add_argument("-mask_num",type=int, default=5000)
    parser.add_argument("-eval_sal",type=str, default="RaCF",help= 'CAM_mode: [RaCF] , [RaCF_GradCAM] , [MC-RISE]')
    parser.add_argument("--run_ins_del",action="store_true",help='True:run False:not run')
    parser.add_argument("--run_adcc",action="store_true",help='True:run False:not run')
    parser.add_argument("--make_imagenet",action="store_true")
    parser.add_argument("-hsv",action="store_true")
    return parser

def parser_demo():
    parser = argparse.ArgumentParser()
    parser.add_argument("-device_num",type=int, default=0,help='GPU device:deep1 [0] or [1]')
    parser.add_argument("-data",type=str, default="GTSRB",help= 'dataset use [GTSRB] or [ImageNet]')
    parser.add_argument("--train",action="store_true",help='GTSRB')
    parser.add_argument("-mode",type=str, default="RaCF",help= 'CAM_mode: [RaCF] , [RaCF_GradCAM] , [MC-RISE] , [eval]')
    parser.add_argument("--make_mask",action="store_true")
    parser.add_argument("-mask_num",type=int, default=5000)
    parser.add_argument("-eval_sal",type=str, default="RaCF",help= 'CAM_mode: [RaCF] , [RaCF_GradCAM] , [MC-RISE]')
    parser.add_argument("--run_ins_del",action="store_true",help='True:run False:not run')
    parser.add_argument("--run_adcc",action="store_true",help='True:run False:not run')
    parser.add_argument("--make_imagenet",action="store_true")
    parser.add_argument("-gtsrb_name",type=str, default="stop.jpg")
    parser.add_argument("-imagenet_split",type=int, default=50)
    parser.add_argument("-hsv",action="store_true")
    return parser

def plot_LossAcc(history,to_path): 
    #lossとsccracyのグラフを保存      
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, label='Training acc',color="b")
    plt.plot(epochs, val_acc, label='Validation acc',color="r")
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig(to_path+"acc.png")
    plt.close()
    plt.figure()

    plt.plot(epochs, loss, label='Training loss',color="b")
    plt.plot(epochs, val_loss, label='Validation loss',color="r")
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(to_path+"loss.png")
    plt.close()
    print('end learn')

def normalization(saliency):
    #各重要度画像に対して正規化を行う
    min_value = saliency.min(axis=(1, 2, 3),keepdims=True)
    max_value = saliency.max(axis=(1, 2, 3),keepdims=True)
    mask = (saliency - min_value) / (max_value - min_value)
    return mask
#チャンネル数を3と想定
def plot_saliency(saliency,image,image_name,result_path):
    saliency = normalization(saliency)
    #pdb.set_trace()
    for i in range(saliency.shape[3]):
        plt.subplot(2,2,i+1)
        plt.imshow(saliency[0,:,:,i],vmin=saliency.min(),vmax=saliency.max(),cmap='jet')
        plt.axis('off')
    plt.subplot(2,2,4)
    plt.imshow(image[0])
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(result_path+f'result_{image_name}.png')
    plt.close()
