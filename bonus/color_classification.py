import os
import cv2
import csv
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


def Img_preprocess(images):
    """
    对输入的图像列表进行预处理
    :images: 图像列表
    :return: 预处理后的图像列表
    """
    preprocessed_images = []

    for image in images:
        # 调整图像大小为48*48
        resized_image = cv2.resize(image, (48, 48))
        height, width = resized_image.shape[:2]
        keep_pixels = int(0.15 * min(height, width))
        cropped_image = resized_image[keep_pixels:-keep_pixels, keep_pixels:-keep_pixels]
        preprocessed_images.append(cropped_image)

    print("Img_preprocess End\n")
    return preprocessed_images

def color_feature(X):
    '''
    将RGB图片转换成HSV图片，并计算HSV空间下的颜色直方图
    :param X: RGB img集合
    :return: HSV空间下的颜色直方图特征集合
    '''
    hsv_features = []
    for img in X:
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 计算H、S、V通道的颜色直方图
        h_hist = cv2.calcHist([hsv_img], [0], None, [180], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv_img], [1], None, [180], [0, 180]).flatten()
        v_hist = cv2.calcHist([hsv_img], [2], None, [256], [0, 256]).flatten()

        # 归一化
        h_hist = h_hist / np.sum(h_hist) if np.sum(h_hist) != 0 else h_hist
        s_hist = s_hist / np.sum(s_hist) if np.sum(s_hist) != 0 else s_hist
        v_hist = v_hist / np.sum(v_hist) if np.sum(v_hist) != 0 else v_hist

        # 组合H、S、V通道的直方图特征
        hsv_feature = np.concatenate([h_hist, s_hist, v_hist])
        hsv_features.append(hsv_feature)

    print("HSV_feature End\n")
    return hsv_features

def train_and_test(path, test_path, csv_path):

#-------------------------------Train-------------------------------#


    X = []  # img数据集
    Y = []  # label数据集

    for class_name in os.listdir(path): # 遍历path下的所有文件夹
        class_path = os.path.join(path, class_name) # 将当前文件夹名称和path拼接，得到img_path

        label = None
        if class_name in ['0', '1', '2', '3', '4', '5', '7', '8', '14', '15', '17', '9', '10', '16', '11', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '13']:
            label = '0'
        elif class_name in ['6', '32', '41', '42', '12']:
            label = '1'
        elif class_name in ['33', '34', '35', '36', '37', '38', '39', '40']:
            label = '2'
        else:
            label = '-1'  # 根据需要设置未知标签的值

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            X.append(img)
            Y.append(label)

    X_prepro = Img_preprocess(X) # 这里到底是30%
    X_color = color_feature(X_prepro)
    color_clf = RandomForestClassifier()
    color_clf.fit(X_color, Y)

    model_path = 'D:\\FL\\color_classifier.pkl' 
    with open(model_path, 'wb') as f:
        pickle.dump(color_clf, f)
    
    print("Train End\n")
    

#-------------------------------Test-------------------------------#

# def test(test_path, csv_path):
    X_test = []  # img数据集
    Y_test = []  # img标签集

    # 从csv文件中加载路径-图像-标签
    with open(csv_path, 'r') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            real_path = row['Path'].replace('Test/', 'Roi_test/')
            img_path = os.path.join(test_path, real_path)
            img = cv2.imread(img_path, 1)
            X_test.append(img)
            Y_test.append(row['ColorId'])
           
    
    test_prepro = Img_preprocess(X_test)
    test_feature = color_feature(test_prepro)
    accuracy = color_clf.score(test_feature, Y_test)
    print("The color classification accuracy is:",  accuracy)

    Y_pred = color_clf.predict(test_feature)
    contrix = confusion_matrix(Y_test, Y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(contrix, annot=True, cmap="YlGnBu", fmt="d", xticklabels=color_clf.classes_, yticklabels=color_clf.classes_)

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()



if __name__ == "__main__":
    path = 'D:\\Github\\Traffic_Sign_Recognition\\Dataset\\Roi_train'
    test_path = "D:\\Github\\Traffic_Sign_Recognition\\Dataset"
    csv_path = "D:\\Github\\Traffic_Sign_Recognition\\Dataset\\Test.csv"
    train_and_test(path, test_path, csv_path)


    


# # 调用函数加载测试数据集
# test_path = '指定测试图像路径'
# csv_path = '指定CSV文件路径'
# X_test, Y_test = load_test_data(test_path, csv_path)


