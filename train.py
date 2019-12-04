from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import time
import dlib
import os

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report
from joblib import dump, load

os.environ["CUDA_VISIBLE_DEVICES"]= "0"

def load_pre_model():
    
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor('model_dlib/shape_predictor_5_face_landmarks.dat')
    facerec = dlib.face_recognition_model_v1('model_dlib/dlib_face_recognition_resnet_model_v1.dat')

    return detector, sp, facerec

def load_data(path_group):

    detector, sp, facerec = load_pre_model()

    # path_ = 'data_asilla/'
    train_y = []
    train_x = []
    for index, name in enumerate(os.listdir(path_group)):
        try:
            path_member = path_group + name + '/' 
            count_100 = 0 # so anh detect được mặt quắ bé
            count_1000 = 0 # số ảnh k đọc được
            count_10000 = len(os.listdir(path_member))
            for path_img in os.listdir(path_member):

                img = cv2.imread(path_member + path_img, 1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                faces = detector(img, 1)
                
                if len(faces) > 0:
                    for face in faces:
                        x = face.left()
                        y = face.top()
                        w = face.right()
                        h = face.bottom()

                        img_ = img[y:h, x:w, :]

                    if img_.shape[0] < 70: 
                        count_100 += 1
                        continue

                    try:              
                        shape = sp(img, face)
                        face_descriptor = facerec.compute_face_descriptor(img, shape)

                        train_x.append(face_descriptor)
                        
                        train_y.append(name)
                    except:
                        count_1000 += 1
                        continue

        except:
            count_1000 += 1
            continue

        print(name)
        print('count face small: ', count_100)# so anh detect được mặt quắ bé
        print("count face can't detect: ", count_1000)# số ảnh k đọc được
        print("good image: ", count_10000 - count_1000 - count_100)
        print()

    return train_x, train_y

def train(train_x, train_y, test_size):

    X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=test_size, random_state=43)

    params_grid = [{'kernel': ['rbf'], 
                    'gamma': [1e-3, 1e-4],
                    'C': [1, 10, 100, 1000],
                    'probability':[True]
                    },
                    {'kernel': ['linear'], 
                    'C': [1, 10, 100, 1000],
                    'probability':[True]
                    }]

    svm_model = GridSearchCV(SVC(), params_grid, cv=5)
    svm_model.fit(X_train, y_train)

    final_model = svm_model.best_estimator_
    dump(final_model, 'final_model3.joblib') 

    y_pred = final_model.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print("\n")
    print(classification_report(y_test, y_pred))

def main():
    
    train_x, train_y = load_data('data/')
    train(train_x, train_y, 0)

if __name__ == '__main__': 
    main()
    

