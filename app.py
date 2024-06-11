import os
from datetime import date

import cv2
from flask import Flask, render_template, request

from pyattendance import Attendance, FaceDetection

#### Defining Flask App
app = Flask(__name__)

today = date.today().strftime("%d-%B-%Y")

atd = Attendance()
face_detector = FaceDetection()

################## ROUTING FUNCTIONS #########################


#### Our main page
@app.route("/")
def home():
    names, rolls, times, l = atd.extract_attendance()
    return render_template(
        "home.html",
        names=names,
        rolls=rolls,
        times=times,
        l=l,
        totalreg=atd.totalreg(),
        today=today,
    )


#### This function will run when we click on Take Attendance Button
@app.route("/start", methods=["GET"])
def start():
    if "face_recognition_model.pkl" not in os.listdir("static"):
        return render_template(
            "home.html",
            totalreg=atd.totalreg(),
            today=atd.today,
            mess="There is no trained model in the static folder. Please add a new face to continue.",
        )

    cap = cv2.VideoCapture(0)
    ret = True
    face_count = 0
    while ret:
        ret, frame = cap.read()
        if face_detector.is_face_and_eye(frame):

            identified_person, frame = face_detector.detect(frame)

            atd.add_attendance(identified_person)
            cv2.putText(
                frame,
                f"{identified_person}",
                (200, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 20),
                2,
                cv2.LINE_AA,
            )
            face_count += 1

        elif face_detector.is_eye(frame):
            identified_person, frame = face_detector.detect(frame, only_eyes=True)

            atd.add_attendance(identified_person)
            cv2.putText(
                frame,
                f"{identified_person}",
                (200, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 20),
                2,
                cv2.LINE_AA,
            )
            face_count += 1

        cv2.putText(
            frame,
            f"Faces: {face_count}/20",
            (35, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 20),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Attendance", frame)
        if (cv2.waitKey(1) & 0xFF == ord("q")) or face_count == 20:
            break

    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = atd.extract_attendance()
    return render_template(
        "home.html",
        names=names,
        rolls=rolls,
        times=times,
        l=l,
        totalreg=atd.totalreg(),
        today=today,
    )


#### This function will run when we add a new user
@app.route("/add", methods=["GET", "POST"])
def add():
    newusername = request.form["newusername"].strip()
    newuserid = request.form["newuserid"].strip()
    userfacefolder = "static/faces/" + newusername + "_" + str(newuserid)
    usereyefolder = "static/eyes/" + newusername + "_" + str(newuserid)

    if not os.path.isdir(userfacefolder):
        os.makedirs(userfacefolder)
    if not os.path.isdir(usereyefolder):
        os.makedirs(usereyefolder)

    cap = cv2.VideoCapture(0)
    n_face_caps, n_faces = 0, 0
    n_eyes = 0
    faces_captured = False
    eyes_captured = False
    while 1:
        _, frame = cap.read()
        faces = face_detector.extract_faces(frame)
        if len(faces) and not (faces_captured):
            for x, y, w, h in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                cv2.putText(
                    frame,
                    f"Face Images Captured: {n_face_caps}/50",
                    (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 20),
                    2,
                    cv2.LINE_AA,
                )
                if n_faces % 5 == 0:
                    name = newusername + "_" + str(n_face_caps) + ".jpg"
                    cv2.imwrite(
                        userfacefolder + "/" + name, frame[y : y + h, x : x + w]
                    )
                    n_face_caps += 1
                n_faces += 1

        eyes = face_detector.extract_eyes(frame)

        if (len(eyes) == 2) and not (eyes_captured):
            for i, eye in enumerate(eyes):

                (x, y, w, h) = eye
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                cv2.putText(
                    frame,
                    f"Eye Images Captured: {n_eyes//5}/50",
                    (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 20),
                    2,
                    cv2.LINE_AA,
                )
                if n_eyes % 5 == 0:
                    name = newusername + "_" + str(n_eyes // 5) + "_" + str(i) + ".jpg"
                    cv2.imwrite(usereyefolder + "/" + name, frame[y : y + h, x : x + w])
            n_eyes += 1

        if n_faces == 250:
            faces_captured = True

        if n_eyes == 250:
            eyes_captured = True

        if (n_faces == 250) and (n_eyes == 250):
            break

        cv2.imshow("Adding new User", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Training Model")
    face_detector.train_model()
    names, rolls, times, l = atd.extract_attendance()
    return render_template(
        "home.html",
        names=names,
        rolls=rolls,
        times=times,
        l=l,
        totalreg=atd.totalreg(),
        today=today,
    )


#### Our main function which runs the Flask App
if __name__ == "__main__":
    app.run(debug=True)
