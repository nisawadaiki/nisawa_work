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
parser = parser_run()
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
    img_path = path+'images'
    #マスクのパス
    mask_path = path+f'mask{N}/'
    os.makedirs(mask_path,exist_ok=True)

    layer_name='conv2d_12'

    #モデルの呼び出し
    model = vgg16(input_size,class_num)
    #マスクの作成
    if generate_new:
        generate_masks(mask_path,IMAGE_SIZE,N=N, s=s, p1=p_mask, savepath=maskname)

    #GTSRBテストのデータ数を決める(全部で10000枚あるため、データ数を少なくする)
    data_num=1000
    #分類に成功した画像のみを選択
    if opt.train==False:
        if opt.hsv:
            print('HSV')
            model_name = 'vgg16_sgd_hsv_weights.h5'
            model.load_weights(checkpoint_path+model_name)
            test_images,test_labels = correct_hsv_data(model,data_num,IMAGE_SIZE)
        else:
            print('RGB')
            model_name = 'vgg16_sgd_weights.h5'
            model.load_weights(checkpoint_path+model_name)
            test_images,test_labels = correct_data(model,data_num,IMAGE_SIZE)
        
elif opt.data == 'ImageNet':
    from ImageNet.make_data import *
    from ImageNet.evaluate import *
    IMAGE_SIZE=224
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

    #初めて動作させるときはTrue：pickleファイルがあるなら指定しなくてよい(RGB->BGRで中心化されていることに注意)
    if opt.make_imagenet:
        make_imagenet_data(IMAGE_SIZE, input_file_path, label_file, image_dir, pickle_path)
    
    with open(pickle_path+"data.pickle","rb") as aa:
        images=pickle.load(aa)
    with open(pickle_path+"labels.pickle","rb") as aa:
        labels=pickle.load(aa)

    #マスクの作成
    if generate_new:
        generate_masks(mask_path,IMAGE_SIZE,N=N, s=s, p1=p_mask, savepath=maskname)
    
    use_labels = np.array([1,40,277,402,404,499,566,817,919,943,945,949,950,951,954])
    test_images,test_labels = make_data_15class(images, use_labels, labels, model)
    test_images,test_labels = test_images[0:2],test_labels[0:2]

#学習(GTSRBの時のみ)
if opt.train:
    #学習データ作成
    if opt.hsv:
        data = load_hsv_data(img_path,class_num,IMAGE_SIZE)
    else:
        data=load_data(img_path,class_num,IMAGE_SIZE)
    # dataから画像とラベルを分割し、それぞれをNumPy配列に変換
    images = np.array([image for image, label in data])
    labels = np.array([label for image, label in data])
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
    X_train = X_train/255.0
    X_val = X_val/255.0

    # SGDオプティマイザの設定
    learning_rate = 1e-1
    momentum = 0.9
    weight_decay = 5e-4
    sgd = SGD(learning_rate=learning_rate, momentum=momentum, decay=weight_decay)

    # 学習率のスケジューリング
    def lr_scheduler(epoch,lr):
        if epoch % 30 == 0 and epoch > 0:
            lr = lr * 0.1
        return lr

    model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path+model_name, monitor='val_loss', save_weights_only=False, mode='min', save_best_only=True, save_frech='epoch',verbose=1)
    cp_early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min', min_delta=0.0001, verbose=1)
    history = model.fit(X_train,y_train,epochs=90, 
                        batch_size=400,validation_data=(X_val,y_val),shuffle=False,
                        callbacks=[lr_callback,cp_callback,cp_early_stopping],verbose=1)

    plot_LossAcc(history,to_path)

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
    maps = saliency.make_saliency(result_path,save_pickle=True)

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
    maps = saliency.make_gradcam_saliency(result_path,save_pickle=True)

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
    maps = saliency.make_saliency(result_path,save_pickle=True)
    #それぞれの色で出力、保存
    #sal = make_color_saliency(model,explainer,result_path,save_pickle=True)

#各重要度マップを評価
if opt.mode == 'eval':    
    if opt.eval_sal=='MC-RISE':
        result_path = to_path+f'mc_rise/mask_num{N}/'
        from method.MC_RISE import *
        explainer = Mc_Rise(model,opt.data)
    if opt.eval_sal=='RaCF':
        result_path = to_path+f'racf/mask_num{N}/'
        from method.RaCF import *
        explainer = Racf(model,opt.data)
    if opt.eval_sal=='RaCF_GradCAM':
        from method.RaCFplusGradCAM import *
        result_path = to_path+f'gradcam_racf/mask_num{N}/'
        explainer = RaCF_GradCAM(model,opt.data,layer_name)

    with open(result_path+f"saliency{N}.pickle","rb") as aa:
        saliency=pickle.load(aa)
    
    name = '_'

    #insertion,deletion実行
    if opt.run_ins_del:
        ins_del = Insertion_Deletion(saliency,test_images,model,test_labels,N)
        ins_del.insertion_deletion_run(result_path,name)
    #adccを実行
    if opt.run_adcc:
        adcc = Adcc(saliency,explainer,model,test_images,test_labels,maskname,p_mask,N)
        adcc.adcc_run(opt.mode,result_path,mask_path)
