
import pandas as pd
import cv2
import numpy as np
import os


def load_data(image_path,classes,image_resize):
    data = []
    for i in range(classes):
        path = os.path.join(image_path,'Train',str(i))
        images = os.listdir(path)
        for image_name in images:
            try:
                image = cv2.imread(path + '/'+ image_name)
                image = cv2.resize(image,(image_resize,image_resize))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = np.array(image)
                data.append([image,i])
            except:
                print("Error loading image")
    return data

def load_hsv_data(image_path,classes,image_resize):
    data = []
    for i in range(classes):
        path = os.path.join(image_path,'Train',str(i))
        images = os.listdir(path)
        for image_name in images:
            try:
                image = cv2.imread(path + '/'+ image_name)
                image = cv2.resize(image,(image_resize,image_resize))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                image = np.array(image)
                data.append([image,i])
            except:
                print("Error loading image")
    return data


def load_test_data(image_resize,csv_path='/data1/nisawa/gtsrb/image/Test.csv'):
    y_test = pd.read_csv(csv_path)
    labels = y_test["ClassId"].values
    imgs = y_test["Path"].values
    
    data=[]
    for img in imgs:
        image = cv2.imread('/data1/nisawa/gtsrb/image/'+img)
        image = cv2.resize(image,(image_resize,image_resize))
        data.append(np.array(image))

    data = np.array(data)
    return data,labels

def currnnt_data(model, data_num, image_resize):
    data,labels = load_test_data(image_resize)
    test_images = np.zeros((data.shape[0],data.shape[1],data.shape[2],data.shape[3]))
    for i in range(data.shape[0]):
        rgb = cv2.cvtColor(data[i], cv2.COLOR_BGR2RGB)
        test_images[i] = rgb/255.

    predict=model.predict(test_images[:data_num])
    predict_label=(np.argmax(predict,axis=-1))
    # 誤ったラベルを除外
    correct_indices = (predict_label == labels[:data_num])
    model_outputs = predict_label[correct_indices]
    true_labels = labels[:data_num][correct_indices]
    test_images=test_images[:data_num][correct_indices]
    return test_images,true_labels
