# image processing
import cv2
# matrix and maths
import numpy as np
# deep learning and 68 landmarks points detector
import dlib
# face utils for basic operation
from imutils import face_utils
from pygame import mixer

mixer.init()
sound_1 = mixer.Sound('alarm.wav')
sound_2 = mixer.Sound('alert.wav')

# initializing the camera-------------
cap = cv2.VideoCapture(0)

# initializing the face detector and landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# status marking for current state
sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)


def compute(ptA, ptB):
    dist = np.linalg.norm(ptA-ptB)
    return dist

# blinking status


def blinked(a, b, c, d, e, f):
    new_var = compute(b, d)
    print(new_var)
    up = compute(b, d) + compute(c, e)  # (37,41)(38,40)
    down = compute(a, f)
    ratio = up/(2.0*down)

    # check if it is blinked or not ?....
    if (ratio > 0.22):  # eyes open
        return 2
    elif(ratio <= 0.22 and ratio > 0.17):  # eyes browsy
        return 1
    else:  # eyes closed
        return 0


while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        face_frame = frame.copy()
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # landmarks-----------------
        landmark = predictor(gray, face)
        landmark = face_utils.shape_to_np(landmark)

        # access landmarks parts
        left_blink = blinked(
            landmark[36], landmark[37], landmark[38], landmark[41], landmark[40], landmark[39])
        right_blink = blinked(
            landmark[42], landmark[43], landmark[44], landmark[47], landmark[46], landmark[45])

        # show eye blink status----------
        if left_blink == 0 or right_blink == 0:
            sleep += 1
            drowsy = 0
            active = 0
            if sleep > 6:
                status = "sleeping"
                color = (255, 0, 0)
                sound_2.play()
        elif left_blink == 1 or right_blink == 1:
            sleep = 0
            drowsy += 1
            active = 0
            if drowsy > 6:
                status = "drowsy !!!"
                color = (0, 0, 255)
                sound_1.play()
        else:
            sleep = 0
            drowsy = 0
            active += 1
            if active > 6:
                status = "active ..."
                color = (0, 255, 0)

        # put and show text
        cv2.putText(frame, status, (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # show all landmark points
        for n in range(0, 68):
            (x, y) = landmark[n]
            cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

    cv2.imshow("frame", frame)
    cv2.imshow("face_frame", face_frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
