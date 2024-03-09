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
    parser.add_argument("--hsv",action="store_true")
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
    parser.add_argument("--hsv",action="store_true")
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
    #サブプロットを作成
    fig, axs = plt.subplots(1, 4,figsize=(12, 3))
    # プロットを設定
    axs[0].imshow(saliency[0,:,:,0],vmin=saliency.min(),vmax=saliency.max(),cmap='jet')
    axs[0].set_title('red')
    axs[1].imshow(saliency[0,:,:,1],vmin=saliency.min(),vmax=saliency.max(),cmap='jet')
    axs[1].set_title('green')
    axs[2].imshow(saliency[0,:,:,2],vmin=saliency.min(),vmax=saliency.max(),cmap='jet')
    axs[2].set_title('blue')
    axs[3].imshow(image[0])
    axs[3].set_title('image')

    # レイアウトを調整
    plt.tight_layout()
    plt.savefig(result_path+f'result_{image_name}.png')
    plt.close()

def plot_color_saliency(saliency,image,image_name,result_path):
    saliency = np.max(saliency,axis=-1)
    saliency = normalization(saliency)
    #pdb.set_trace()
    #サブプロットを作成
    fig, axs = plt.subplots(3, 3,figsize=(12, 12))
    # プロットを設定
    axs[0,0].imshow(saliency[0,0,:,:],vmin=saliency.min(),vmax=saliency.max(),cmap='jet')
    axs[0,0].set_title('red')
    axs[0,1].imshow(saliency[0,1,:,:],vmin=saliency.min(),vmax=saliency.max(),cmap='jet')
    axs[0,1].set_title('green')
    axs[0,2].imshow(saliency[0,2,:,:],vmin=saliency.min(),vmax=saliency.max(),cmap='jet')
    axs[0,2].set_title('blue')
    axs[1,0].imshow(saliency[0,3,:,:],vmin=saliency.min(),vmax=saliency.max(),cmap='jet')
    axs[1,0].set_title('yellow')
    axs[1,1].imshow(saliency[0,4,:,:],vmin=saliency.min(),vmax=saliency.max(),cmap='jet')
    axs[1,1].set_title('purple')
    axs[1,2].imshow(saliency[0,5,:,:],vmin=saliency.min(),vmax=saliency.max(),cmap='jet')
    axs[1,2].set_title('light_blue')
    axs[2,0].imshow(saliency[0,6,:,:],vmin=saliency.min(),vmax=saliency.max(),cmap='jet')
    axs[2,0].set_title('white')
    axs[2,1].imshow(image[0])
    axs[2,1].set_title('image')

    # レイアウトを調整
    plt.tight_layout()
    plt.savefig(result_path+f'result_{image_name}.png')
    plt.close()
