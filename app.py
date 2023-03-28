import cv2
import os
from flask import Flask,request,render_template
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

#### Defining Flask App
app = Flask(__name__)


#### Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")


#### Initializing VideoCapture object to access WebCam
# for front face
face_detector = cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml')
eye_detector = cv2.CascadeClassifier('static/haarcascade_eye_tree_eyeglasses.xml')



#### If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if not os.path.isdir('static/eyes'):
    os.makedirs('static/eyes')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv','w') as f:
        f.write('Name,Roll,Time')


def create_features(face, eye_1, eye_2):
    face = cv2.resize(face, (50, 50))
    eye_1 = cv2.resize(eye_1, (10, 10))
    eye_2 = cv2.resize(eye_2, (10, 10))

    features = np.hstack((face.ravel(), eye_1.ravel(), eye_2.ravel()))
    return features

#### get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))

#### extract the face from an image
def extract_eyes(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eye_points = eye_detector.detectMultiScale(gray, 1.3, 5)
    return eye_points

#### extract the face from an image
def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray, 1.3, 5)
    return face_points


#### Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


#### A function which trains the model on all the faces available in faces folder
def train_model():
    features = []
    #eyes = []
    labels = []
    userlist = os.listdir('static/faces')

    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            face = cv2.resize(img, (50, 50))

            eye_1 = cv2.imread(f'static/eyes/{user}/{imgname[: -4]}_0.jpg')
            eye_2 = cv2.imread(f'static/eyes/{user}/{imgname[: -4]}_1.jpg')
            
            labels.append(user)
            features.append(create_features(face, eye_1, eye_2))

    features = np.array(features)
    knn = KNeighborsClassifier(n_neighbors = 5)
    knn.fit(features, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')


#### Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names,rolls,times,l


#### Add Attendance of a specific user
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv','a') as f:
            f.write(f'\n{username},{userid},{current_time}')


################## ROUTING FUNCTIONS #########################

#### Our main page
@app.route('/')
def home():
    names,rolls,times,l = extract_attendance()    
    return render_template('home.html',names=names,rolls=rolls,times=times,l=l,totalreg=totalreg(),datetoday2=datetoday2) 


#### This function will run when we click on Take Attendance Button
@app.route('/start',methods=['GET'])
def start():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html',totalreg=totalreg(),datetoday2=datetoday2,mess='There is no trained model in the static folder. Please add a new face to continue.') 

    cap = cv2.VideoCapture(0)
    ret = True
    face_count = 0
    while ret:
        ret,frame = cap.read()
        if np.any(extract_faces(frame)) and (extract_eyes(frame).shape[0] == 2):
            (x,y,w,h) = extract_faces(frame)[0]
            cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
            face = frame[y:y+h,x:x+w]

            eye_1_coords, eye_2_coords = extract_eyes(frame)
            (x,y,w,h) = eye_1_coords
            cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
            eye_1 = frame[y:y+h,x:x+w]

            (x,y,w,h) = eye_2_coords
            cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
            eye_2 = frame[y:y+h,x:x+w]

            features = create_features(face, eye_1, eye_2)[np.newaxis, ...]

            identified_person = identify_face(features)[0]
            add_attendance(identified_person)
            cv2.putText(frame,f'{identified_person}',(200,35),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
            face_count+=1

        cv2.putText(frame,f'Faces: {face_count}/20',(35,35),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
        cv2.imshow('Attendance',frame)
        if (cv2.waitKey(1) & 0xFF == ord('q')) or face_count == 20:
            break
    cap.release()
    cv2.destroyAllWindows()
    names,rolls,times,l = extract_attendance()    
    return render_template('home.html',names=names,rolls=rolls,times=times,l=l,totalreg=totalreg(),datetoday2=datetoday2) 


#### This function will run when we add a new user
@app.route('/add',methods=['GET','POST'])
def add():
    newusername = request.form['newusername'].strip()
    newuserid = request.form['newuserid'].strip()
    userfacefolder = 'static/faces/'+newusername+'_'+str(newuserid)
    usereyefolder = 'static/eyes/'+newusername+'_'+str(newuserid)

    if not os.path.isdir(userfacefolder):
        os.makedirs(userfacefolder)
    if not os.path.isdir(usereyefolder):
        os.makedirs(usereyefolder)

    
    cap = cv2.VideoCapture(0)
    n_face_caps, n_faces = 0, 0
    n_eye_caps, n_eyes = 0, 0
    faces_captured = False
    eyes_captured = False
    while 1:
        _,frame = cap.read()
        faces = extract_faces(frame)
        if len(faces) and not(faces_captured):
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x, y), (x + w, y + h), (255, 0, 20), 2)
                cv2.putText(frame,f'Face Images Captured: {n_face_caps}/50',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
                if n_faces % 5==0:
                    name = newusername+'_'+str(n_face_caps)+'.jpg'
                    cv2.imwrite(userfacefolder + '/' + name, frame[y: y + h, x: x + w])
                    n_face_caps+=1
                n_faces+=1

        eyes = extract_eyes(frame)
        
        if (len(eyes) == 2) and not(eyes_captured):
            for i, eye in enumerate(eyes):
                
                (x,y,w,h) = eye
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                cv2.putText(frame,f'Eye Images Captured: {n_eyes//5}/50', (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 20),2,cv2.LINE_AA)
                if n_eyes%5==0:
                    name = newusername + '_' + str(n_eyes//5) + '_' + str(i) + '.jpg'
                    cv2.imwrite(usereyefolder + '/' + name, frame[y: y + h, x: x + w])

            n_eyes+=1
        

        if (n_faces == 250):
            faces_captured = True

        if (n_eyes == 250):
            eyes_captured = True
        
        if (n_faces == 250) and (n_eyes == 250):
            break

        cv2.imshow('Adding new User',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names,rolls,times,l = extract_attendance()    
    return render_template('home.html',names=names,rolls=rolls,times=times,l=l,totalreg=totalreg(),datetoday2=datetoday2) 


#### Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)