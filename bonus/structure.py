import os.path
import cv2
import joblib
from skimage.feature import hog, daisy, canny
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from skimage.filters import sobel
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import time
import pickle
import csv
from pathlib import Path
import glob
from color_classification import color_feature

signClassPath0 = 'D:\\FL\\0_clf.pickle'
signClassPath1 = 'D:\\FL\\1_clf.pickle'
signClassPath2 = 'D:\\FL\\2_clf.pickle'

def load_train(path):
    X = []
    Y = []

    for dir_name in os.listdir(path):
        img_path = os.path.join(path, dir_name)
        for img_file in glob.glob(os.path.join(img_path, '*.png')):
            label = dir_name  # 将文件夹名作为标签
            Y.append(label)

            img = cv2.imread(img_file, 1)  # 加载彩色图像
            X.append(img)

    return X, Y

def Img_preprocess(images):
    preprocessed_images = []

    for img in images:
        height, width = img.shape[:2]
    
        new_width = int(width * 0.8)
        new_height = int(height * 0.8)
        left = (width - new_width) // 2
        top = (height - new_height) // 2
        right = left + new_width
        bottom = top + new_height
        cropped_img = img[top:bottom, left:right] # 对应load_train
       
        resized_img = cv2.resize(cropped_img, (48, 48)) # 1. 统一大小
        gray_img = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY) # 2. 灰度化处理
        equalized_img = cv2.equalizeHist(gray_img) # 3. Hist直方图

        preprocessed_images.append(equalized_img)

    return preprocessed_images

def Extract_feature(images): # 已改sob 这里的逻辑可能要改
    X_hog = []
    X_sob = []

    for x in images:
        fe_hog = hog(x, orientations=8, pixels_per_cell=(10, 10),
                    cells_per_block=(1, 1), visualize=False)
        X_hog.append(fe_hog)
        fe_sob = sobel(x)
        X_sob.append(fe_sob)
    X_sob_flat = [feature.flatten() for feature in X_sob]

    X_feature = np.concatenate((X_hog, X_sob_flat), axis=1) # 到底是0还是1
    return X_feature

def load_test(test_path, csv_path): # 仅读取 原始图像
    X1 = []  # 原始图像数据集
    Y = []   # 图像标签集

    with open(csv_path, 'r', errors='ignore') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            label = row['ClassId']
            real_path = row['Path'].replace('Test/', 'Roi_test/')
            img_path = os.path.join(test_path, real_path)
            img = cv2.imread(img_path, 1)
            
            # X1.append(image)
            X1.append(img)
            Y.append(label)

    return  X1, Y 

def load_models(model_paths):
    # sign classifiers
    models = {}
    with open(model_paths['sign_clf0'], 'rb') as f:
        models['sign_clf0'] = pickle.load(f)
    with open(model_paths['sign_clf1'], 'rb') as f:
        models['sign_clf1'] = pickle.load(f)
    with open(model_paths['sign_clf2'], 'rb') as f:
        models['sign_clf2'] = pickle.load(f)
    return models

def color_predict(X_ori, model_paths): # 对中间70%的图片提取颜色特征进行颜色预测
   
    X_prepro = []
    for x in X_ori: # 只保留中间70%
        x_re = cv2.resize(x, (48,48))
        height, width = x_re.shape[:2]
        keep_pixels = int(0.15 * min(height, width))
        x_crop = x_re[keep_pixels:-keep_pixels, keep_pixels:-keep_pixels]
        X_prepro.append(x_crop)
    
    X_features = color_feature(X_prepro) 
    with open(model_paths['color_classifier'], 'rb') as f:
        col_clf = pickle.load(f)
    X_color = col_clf.predict(X_features)

    return X_color

def sign_predict(X_ori, X_color, sign_classifiers):
    # Predict signs based on features and color
    X2_prepro = Img_preprocess(X_ori)
    X2_features = Extract_feature(X2_prepro)
    X_combined = np.concatenate((X_color.reshape(-1, 1), X2_features), axis=1)

    X_sign = []
    for x in X_combined:
        color = int(x[0])  # Assuming X_color is an integer
        x_feature = x[1:].reshape(1, -1)
        if color == 0:
            sign_clf = sign_classifiers['sign_clf0']
        elif color == 1:
            sign_clf = sign_classifiers['sign_clf1']
        else:
            sign_clf = sign_classifiers['sign_clf2']

        x_predict = sign_clf.predict(x_feature)
        X_sign.append(str(x_predict[0]))

    return X_sign

def Train(train_path, model_paths):
    for color in [0, 1, 2]:
        print('Train color_set:', color)
        colorSetPath = os.path.join(train_path, str(color))
        X, Y = load_train(colorSetPath)
        X_prepro = Img_preprocess(X)
        X_features = Extract_feature(X_prepro)
        classifier = SVC(C=100, gamma=0.1, kernel='rbf')  # 初始化分类器
        classifier.fit(X_features, Y)
       
        model_file = model_paths[f'sign_clf{color}']
        with open(model_file, 'wb') as f:
            pickle.dump(classifier, f)
        # LoadANDSave.model_save(classifier, os.path.join('classifuModels', str(color), 'clf.pickle'))

def Test(test_path, csv_path, model_paths):
    start_time = time.time()

    #col_X, sign_X, Y = load_test(test_path, csv_path) # 这里其实只读取了图片，没有对图像做处理
    X_ori, Y = load_test(test_path, csv_path)
    X_color = color_predict(X_ori, model_paths) 
## al

#----------------------Color Prediction End----------------------#

    sign_classifiers = load_models(model_paths) # 加载SVM分类器
    # 调试
    print("sign classifier load end")

    X_predict = sign_predict(X_ori, X_color, sign_classifiers) # 这个函数里面需要进行对X_ori裁剪的部分
    # models['color_classifier']

#---------------------- Sign Prediction End ---------------------#
    
    correct_predictions = 0
    total_samples = len(Y)

    # Compare predicted signs with true labels
    for i in range(total_samples):
        if X_predict[i] == Y[i]:
            correct_predictions += 1

    # Calculate accuracy
    accuracy = correct_predictions / total_samples
    print("准确率：", accuracy)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time:", elapsed_time, "seconds")

    # Calculate confusion matrix
    cm = confusion_matrix(Y, X_predict)

    # Visualize confusion matrix
    classes = np.unique(Y)
    plt.figure(dpi=700)
    sns.heatmap(cm, annot=True, cmap="YlGnBu", fmt="d", xticklabels=classes, yticklabels=classes, annot_kws={"fontsize": 4})
    plt.xticks(fontsize=4)
    plt.yticks(fontsize=4)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig('confusionimgsave')
    plt.show()


def main():
    train_path = 'D:\\Github_temp\\bonus_hierarchy\\dataset\\Train'
    test_path = 'D:\\Github\\Traffic_Sign_Recognition\\Dataset'
    csv_path = 'D:\\Github\\Traffic_Sign_Recognition\\Dataset\\Test.csv'
    model_paths = {
        'color_classifier': 'D:\\FL\\color_classifier.pkl',
        'sign_clf0': signClassPath0,
        'sign_clf1': signClassPath1,
        'sign_clf2': signClassPath2
    }
    # Train(train_path,model_paths )
    Test(test_path, csv_path, model_paths)

if __name__ == "__main__":
    main()
