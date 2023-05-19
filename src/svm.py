# We will scale all masks to a 100x100 image size.
# Then, we'll train the SVM on the masks with labels (yes or no)
# This class includes the labeling, training, and prediction of the SVM.

# For training, the images are split into two folders: stalk, not_stalk

import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


flat_data_arr = []  # input
target_arr = []  # output


data_dir = 'data'
for cat, i in enumerate(['stalk', 'not_stalk']):
    print(f'loading : {i}')
    path = os.path.join(data_dir, cat)

    for img in os.listdir(path):
        img_array = imread(os.path.join(path, img))
        img_resized = resize(img_array, (100, 100, 3))
        flat_data_arr.append(img_resized.flatten())
        target_arr.append(i)
        print(f'loaded category:{i} successfully')

flat_data = np.array(flat_data_arr)
target = np.array(target_arr)
df = pd.DataFrame(flat_data)  # dataframe
df['Target'] = target
x = df.iloc[:, :-1]  # input data
y = df.iloc[:, -1]  # output data

# MODEL CONSTRUCTION
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.0001, 0.001, 0.1, 1], 'kernel': ['rbf', 'poly']}
svc = svm.SVC(probability=True)
model = GridSearchCV(svc, param_grid)


# MODEL TRAINING
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=77, stratify=y)
print('Splitted Successfully')
model.fit(x_train, y_train)
print('The Model is trained well with the given images')  # model.best_params_ contains the best parameters obtained from GridSearchCV


# MODEL TESTING
y_pred = model.predict(x_test)
print("The predicted Data is :")
print(y_pred)
print("The actual data is:")
print(np.array(y_test))
print(f"The model is {accuracy_score(y_pred,y_test)*100}% accurate")
