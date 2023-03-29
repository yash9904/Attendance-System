import cv2
import os
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

class attendance:
    def __init__(self):

        self.datetoday = date.today().strftime("%m_%d_%y")
        

        if not os.path.isdir('Attendance'):
            os.makedirs('Attendance')
        if not os.path.isdir('static/faces'):
            os.makedirs('static/faces')
        if not os.path.isdir('static/eyes'):
            os.makedirs('static/eyes')
        if f'Attendance-{self.datetoday}.csv' not in os.listdir('Attendance'):
            with open(f'Attendance/Attendance-{self.datetoday}.csv','w') as f:
                f.write('Name,Roll,Time')

    def extract_attendance(self):
        df = pd.read_csv(f'Attendance/Attendance-{self.datetoday}.csv')
        names = df['Name']
        rolls = df['Roll']
        times = df['Time']
        l = len(df)
        return names,rolls,times,l

    #### Add Attendance of a specific user
    def add_attendance(self, name):
        username = name.split('_')[0]
        userid = name.split('_')[1]
        current_time = datetime.now().strftime("%H:%M:%S")
        
        df = pd.read_csv(f'Attendance/Attendance-{self.datetoday}.csv')
        if int(userid) not in list(df['Roll']):
            with open(f'Attendance/Attendance-{self.datetoday}.csv','a') as f:
                f.write(f'\n{username},{userid},{current_time}')

    def totalreg(self):
        return len(os.listdir('static/faces'))


class face_detection:
    def __init__(self):
        self.face_detector = cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml')
        self.eye_detector = cv2.CascadeClassifier('static/haarcascade_eye_tree_eyeglasses.xml')

    def create_features(self, face, eye_1, eye_2):
        face = cv2.resize(face, (50, 50))
        eye_1 = cv2.resize(eye_1, (10, 10))
        eye_2 = cv2.resize(eye_2, (10, 10))

        features = np.hstack((face.ravel(), eye_1.ravel(), eye_2.ravel()))
        return features
    #### extract the face from an image
    def extract_eyes(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eye_points = self.eye_detector.detectMultiScale(gray, 1.3, 5)
        return eye_points

    #### extract the face from an image
    def extract_faces(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = self.face_detector.detectMultiScale(gray, 1.3, 5)
        return face_points

    #### Identify face using ML model
    def identify_face(self, facearray):
        model = joblib.load('static/face_recognition_model.pkl')
        return model.predict(facearray)

    #### A function which trains the model on all the faces available in faces folder
    def train_model(self):
        features = []
        labels = []
        userlist = os.listdir('static/faces')

        for user in userlist:
            for imgname in os.listdir(f'static/faces/{user}'):
                img = cv2.imread(f'static/faces/{user}/{imgname}')
                face = cv2.resize(img, (50, 50))

                eye_1 = cv2.imread(f'static/eyes/{user}/{imgname[: -4]}_0.jpg')
                eye_2 = cv2.imread(f'static/eyes/{user}/{imgname[: -4]}_1.jpg')
                
                labels.append(user)
                features.append(self.create_features(face, eye_1, eye_2))

        features = np.array(features)
        knn = KNeighborsClassifier(n_neighbors = 5)
        knn.fit(features, labels)
        joblib.dump(knn, 'static/face_recognition_model.pkl')
    
    def detect_face(self, frame):

        (x,y,w,h) = self.extract_faces(frame)[0]
        cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
        face = frame[y:y+h,x:x+w]

        eye_1_coords, eye_2_coords = self.extract_eyes(frame)
        (x,y,w,h) = eye_1_coords
        cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
        eye_1 = frame[y:y+h,x:x+w]

        (x,y,w,h) = eye_2_coords
        cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
        eye_2 = frame[y:y+h,x:x+w]

        features = self.create_features(face, eye_1, eye_2)[np.newaxis, ...]

        identified_person = self.identify_face(features)[0]

        return identified_person, frame


    def is_face(self, frame):
        return True if np.any(self.extract_faces(frame)) and (len(self.extract_eyes(frame)) == 2) else False
