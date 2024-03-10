import pickle
import numpy as np
import pdb
import os
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD
from tqdm import tqdm

from make_mask import *
from make_saliency import *
from util import *


os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

#util.pyを参照
parser = parser_demo()
opt = parser.parse_args()

#============================================================
#使用するGPUの決定
# tensorflow2.xでのGPUの設定
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    # 
    #for k in range(len(physical_devices)):
    k = opt.device_num
    tf.config.set_visible_devices(physical_devices[k], 'GPU')
    tf.config.experimental.set_memory_growth(physical_devices[k], True)
    print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
else:
    print("Not enough GPU hardware devices available")

#============================================================

#モード決定(RaCF,RaCF_GradCAM,MC-RISE)
mode = opt.mode
#True:マスクを新しく作る
generate_new = opt.make_mask

#マスク数(デフォルト5000)
N=opt.mask_num
#初めに小さいマスクを作るときの大きさ
s=8
#p_maskの確率
p_mask=0.5
#マスク名
maskname = 'masks.npy'
#作業フォルダ
path = f'/data1/nisawa/nisawa_works/{opt.data}/'
#結果を保存するpath
to_path = path+'result/'
os.makedirs(to_path,exist_ok=True)

print(f'dataset:{opt.data}')
#GTSRBの時の設定
if opt.data == 'GTSRB':
    from GTSRB.evaluate import *
    from GTSRB.make_data import *
    from GTSRB.vgg16 import *
    #画像の大きさ
    IMAGE_SIZE= 96
    #チャンネル数
    C =3
    input_size = (IMAGE_SIZE,IMAGE_SIZE,C)
    #クラス数
    class_num = 43
    
    #モデル保存のpath
    checkpoint_path = to_path+'checkpoint/'
    os.makedirs(checkpoint_path,exist_ok=True)
    #画像のパス
    img_path = path+'images/'
    #マスクのパス
    mask_path = path+f'mask{N}/'
    os.makedirs(mask_path,exist_ok=True)

    layer_name='conv2d_12'

    #モデルの呼び出し
    model = vgg16(input_size,class_num)

    #マスクの作成
    if generate_new:
        generate_masks(mask_path,IMAGE_SIZE,N=N, s=s, p1=p_mask, savepath=maskname)

    image=cv2.imread(img_path+f'{opt.gtsrb_name}')
    images = cv2.resize(image,(IMAGE_SIZE,IMAGE_SIZE))
    if opt.hsv:
        images = cv2.cvtColor(images, cv2.COLOR_BGR2HSV)
        model_name = 'vgg16_sgd_hsv_weights.h5'
    else:
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
        model_name = 'vgg16_sgd_weights.h5'
    model.load_weights(checkpoint_path+model_name)
    #画像の処理
    images=np.expand_dims(images,axis=0)
    test_images = images/255.
    bb=model(test_images)
    #ラベルがあっていることを確認(stop[14],優先標識[12])
    test_labels=np.argmax(bb,axis=-1)
    image_name = opt.gtsrb_name
    print(test_labels)
    sample = test_images
elif opt.data == 'ImageNet':
    from ImageNet.make_data import *
    from ImageNet.evaluate import *
    IMAGE_SIZE=224
    mean = [103.939, 116.779, 123.68]
    model = tf.keras.applications.resnet50.ResNet50(include_top=True,weights='imagenet')
    
    #pickleデータのパス
    pickle_path = path + 'images/pickle/'
    os.makedirs(pickle_path,exist_ok=True)
    #ラベル情報のファイルパス
    input_file_path= '/data1/nisawa/imagenet/val.txt'
    label_file = "/data1/nisawa/imagenet/val_num.txt"
    #画像が格納されたディレクトリのパス
    image_dir = "/data1/nisawa/imagenet/val_images"

    layer_name='conv5_block3_3_conv'

    #マスクのパス
    mask_path = path+f'mask{N}/'
    os.makedirs(mask_path,exist_ok=True)

    #初めて動作させるときはTrue：pickleファイルができたら指定しなくてよい(RGB->BGRで中心化されていることに注意)
    if opt.make_imagenet:
        make_imagenet_data(IMAGE_SIZE, input_file_path, label_file, image_dir, pickle_path)
    
    with open(pickle_path+"data.pickle","rb") as aa:
        images=pickle.load(aa)
    with open(pickle_path+"labels.pickle","rb") as aa:
        labels=pickle.load(aa)

    #マスクの作成
    if generate_new:
        generate_masks(mask_path,IMAGE_SIZE,N=N, s=s, p1=p_mask, savepath=maskname)
    
    #目で見て決めたものなので、変更する余地はかなりあり　色の偏っているものを見つける？
    use_labels = np.array([1,40,277,402,404,499,566,817,919,943,945,949,950,951,954])
    test_images,test_labels = make_data_15class(images, use_labels, labels, model)
    test_images,test_labels = test_images[opt.imagenet_split:opt.imagenet_split+1],test_labels[opt.imagenet_split:opt.imagenet_split+1]
    image_name = opt.imagenet_split
    sample = test_images+mean
    sample=np.where(sample>255,255,sample)
    sample=np.where(sample<0,0,sample)
    sample = sample[:,:,:,::-1]
    sample = sample /255.


print(f'mode:{opt.mode}')
#RaCFの実行
if opt.mode == 'RaCF' :
    from method.RaCF import *
    explainer = Racf(model,opt.data)
    result_path = to_path+f'racf/mask_num{N}/'
    os.makedirs(result_path,exist_ok=True)

    mask=explainer.load_masks(mask_path,maskname,p1=p_mask)
    print('Masks are loaded.')

    #クラスの呼び出し
    saliency = Saliency(explainer,test_images,test_labels,N)
    #重要度マップを出力、保存
    maps = saliency.make_saliency(result_path,save_pickle=False)

    plot_saliency(maps,sample,image_name,result_path)
    #insertion,deletion実行
    if opt.run_ins_del:
        ins_del = Insertion_Deletion(maps,test_images,model,test_labels,N)
        ins_del.insertion_deletion_run(result_path,image_name,run=False)
    #adccを実行
    if opt.run_adcc:
        adcc = Adcc(maps,explainer,model,test_images,test_labels,maskname,p_mask,N)
        adcc.adcc_run(opt.mode,result_path,mask_path,run=False)

#RaCF+GradCAMの実行、保存
if opt.mode == 'RaCF_GradCAM':
    from method.RaCFplusGradCAM import *
    explainer = RaCF_GradCAM(model,opt.data,layer_name)
    result_path = to_path+f'gradcam_racf/mask_num{N}/'
    os.makedirs(result_path,exist_ok=True)

    #GardCAM
    masks=explainer.load_masks(mask_path,maskname,p1=p_mask)
    print('Masks are loaded.')

    #クラスの呼び出し
    saliency = Saliency(explainer,test_images,test_labels,N)
    #重要度マップを出力、保存
    maps = saliency.make_gradcam_saliency(result_path,save_pickle=False)
    plot_saliency(maps,sample,image_name,result_path)
    #insertion,deletion実行
    if opt.run_ins_del:
        ins_del = Insertion_Deletion(maps,test_images,model,test_labels,N)
        ins_del.insertion_deletion_run(result_path,image_name,run=False)
    #adccを実行
    if opt.run_adcc:
        adcc = Adcc(maps,explainer,model,test_images,test_labels,maskname,p_mask,N)
        adcc.adcc_run(opt.mode,result_path,mask_path,run=False)

#MC-RISEの実行、保存
if opt.mode == 'MC-RISE':
    from method.MC_RISE import *
    explainer = Mc_Rise(model,opt.data)
    result_path = to_path+f'mc_rise/mask_num{N}/'
    os.makedirs(result_path,exist_ok=True)

    mask=explainer.load_masks(mask_path,maskname,p1=p_mask)
    print('Masks are loaded.')
    
    #クラスの呼び出し
    saliency = Saliency(explainer,test_images,test_labels,N)
    #重要度マップを出力、保存
    maps_3ch = saliency.make_saliency(mask_path,save_pickle=False)
    #plot_saliency(maps,test_images,image_name,result_path)
    #それぞれの色で出力、保存
    maps = saliency.make_color_saliency(mask_path,save_pickle=False)
    plot_color_saliency(maps,sample,image_name,result_path)
    #insertion,deletion実行
    if opt.run_ins_del:
        ins_del = Insertion_Deletion(maps_3ch,test_images,model,test_labels,N)
        ins_del.insertion_deletion_run(result_path,image_name,run=False)
    #adccを実行
    if opt.run_adcc:
        adcc = Adcc(maps_3ch,explainer,model,test_images,test_labels,maskname,p_mask,N)
        adcc.adcc_run(opt.mode,result_path,mask_path,run=False)
