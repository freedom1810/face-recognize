from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import time
import dlib
import os
from joblib import dump, load
from datetime import datetime
import threading


os.environ["CUDA_VISIBLE_DEVICES"]= "0"

def load_pre_model():

    final_model = load('final_model3.joblib')
    final_model = None
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor('model_dlib/shape_predictor_5_face_landmarks.dat')
    facerec = dlib.face_recognition_model_v1('model_dlib/dlib_face_recognition_resnet_model_v1.dat')

    return final_model, detector, sp, facerec

def main():
    cap = cv2.VideoCapture(0)

    check_in = []
    check_in_db = []

    final_model, detector, sp, facerec = load_pre_model()

    while(True):
        try:

            start = time.time()

            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            dets = detector(frame, 0)

            name = []
            locates = []
            now = datetime.now()

            if len(dets) > 0:
                for k, d in enumerate(dets):

                    left = d.left()
                    top = d.top()
                    right = d.right()
                    bottom = d.bottom()

                    locates.append((left, top, right, bottom))

                    shape =   sp(frame, d)
                    face_descriptor = facerec.compute_face_descriptor(frame, shape)
                    print(final_model.predict_proba([face_descriptor]))

                    score = max(final_model.predict_proba([face_descriptor])[0])
                    if score > 0.6:
                        name.append(final_model.predict([face_descriptor]))
                    else: 
                        name.append(['unknow'])
                    # name.append(['unknow'])

            if len(locates) > 0:
                for i, locate in enumerate(locates):
                    # org 
                    org = locate[:2]
                    #name 
                    n = name[i][0]
                    frame = cv2.putText(frame, n, org, cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 255, 255) , 2, cv2.LINE_AA) 

            fps = str(int(1/((time.time()- start))))

            frame = cv2.putText(frame, fps, (30,30), cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 255, 255) , 2, cv2.LINE_AA) 

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow('test',frame)
            ch = cv2.waitKey(1)
            if ch == 27: 
                cv2.destroyWindow('test')
                break
        except:
            pass

if __name__ == "__main__":
    # main()
    t = threading.Thread(name='main', target=main)

    t.setDaemon(True)

    t.start()

    t.join()

