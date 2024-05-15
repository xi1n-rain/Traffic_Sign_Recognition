import cv2
import sklearn
import glob
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import random

# T1  start _______________________________________________________________________________
# Read in Dataset

dataset_path = "D:\\vcGroup\images\\"

X = []
y = []

for i in glob.glob(dataset_path + '*.png', recursive=True):
    label = i.split("images")[1][1:4]
    y.append(label)
    image = cv2.imread(i)
    X.append(image)

# T1 end ____________________________________________________________________________________

# T2 start __________________________________________________________________________________
# Preprocessing

X_processed = []
for x in X:
    temp_x = cv2.resize(x, (48, 48))
    gray_x = cv2.cvtColor(temp_x, cv2.COLOR_BGR2GRAY)
    X_processed.append(gray_x)

# T2 end ____________________________________________________________________________________

# T3 start __________________________________________________________________________________
# Feature extraction

X_features = []
for x in X_processed:
    x_feature = hog(x, orientations=8, pixels_per_cell=(10, 10),
                    cells_per_block=(1, 1), visualize=False, multichannel=False)
    X_features.append(x_feature)


# Split training & testing sets using sklearn.model_selection.train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, shuffle=True)

'''test'''
print(len(x_train))
print(len(x_test))
print(len(y_train))
print(len(y_test))
# T3 end ____________________________________________________________________________________

# T4 start __________________________________________________________________________________
# train model
classifier = SVC()  
classifier.fit(x_train, y_train)  

# test accuracy
accuracy = classifier.score(x_test, y_test)
print("准确率：", accuracy)