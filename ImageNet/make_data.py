import pickle
import numpy as np
import pdb
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image

def make_imagenet_data(image_size, input_file_path, label_file, image_dir, pickle_path):
    #testdataを取り出す
    # データとラベルを格納するためのリストを初期化
    data = []
    labels = []

    # 数字のみのテキストファイルのパス
    with open(input_file_path, 'r') as input_file, open(label_file, 'w') as output_file:
        for line in input_file:
            # 行をスペースで分割し、2番目の要素を抽出して書き込み
            parts = line.split()
            if len(parts) >= 2:
                output_file.write(parts[1] + '\n')

    # ラベル情報の読み込み
    with open(label_file, "r") as file:
        lines = file.readlines()
        labels = [int(label) for label in lines]
        

    # 画像ファイルの順序を取得
    image_files = os.listdir(image_dir)
    image_files.sort()  # ファイルの順序をソート
    """
    #サンプル画像を見たい場合
    for i in range(len(image_files)):
        if image_files[i] =='ILSVRC2012_val_00000001.JPEG':
            pdb.set_trace()
    """ 
    # 画像データの読み込み
    for num in range(len(image_files)):
        filename = image_files[num]
        image_path = os.path.join(image_dir, filename)
        image1 = cv2.imread(image_path)
        image1 = cv2.resize(image1,(image_resize,image_resize))
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        x = image.img_to_array(image1)
        x = preprocess_input(x)
        data.append(x)

    # dataとlabelsをNumPy配列に変換
    data = np.array(data)
    labels = np.array(labels)

    with open(pickle_path+"data.pickle","wb") as aa:
        pickle.dump(data, aa,protocol=4)
    with open(pickle_path+"labels.pickle","wb") as aa:
        pickle.dump(labels, aa,protocol=4)

def make_data_15class(images,use_labels,labels,model):
    new_labels = np.array([np.where(labels == i)[0] for i in use_labels])
    correct_label=np.copy(new_labels)
    for i in range(len(use_labels)):
        correct_label[i] = use_labels[i]

    new_labels = new_labels.reshape(-1)
    correct_label = correct_label.reshape(-1)

    new_images = images[new_labels]
    pred=model.predict(new_images)
    pred_label=(np.argmax(pred,axis=-1))
    # 誤ったラベルを除外
    correct_indices = (pred_label == correct_label)
    model_outputs = pred_label[correct_indices]
    true_labels = correct_label[correct_indices]
    test_images=new_images[correct_indices]
    return test_images,true_labels