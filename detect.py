import numpy as np
import cv2
import dlib
from scipy.spatial import distance as dist
from playsound import playsound

#initialize variables
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 3
EAR_CONSEC_FRAMES_CLOSED = 30
EAR_AVG = 0
COUNTER = 0
BLINKS = 0

def eye_aspect_ratio(eye):
    # calculate the euclidean distance between the vertical eye landmarks
    V1 = np.linalg.norm(eye[1] - eye[5])
    V2 = np.linalg.norm(eye[2] - eye[4])

    # calculate the euclidean distance between the horizontal eye landmarks
    H = np.linalg.norm(eye[0] - eye[3])

    # calculate the Eye Aspect Ratio
    EAR = (V1 + V2) / (2 * H)
    return EAR

# to detect the facial region
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# capture video from live video stream
cap = cv2.VideoCapture(0)
while True:
    # get the frame
    ret, frame = cap.read()
    #frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    if ret:
        # convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        for rect in rects:
            x = rect.left()
            y = rect.top()
            x1 = rect.right()
            y1 = rect.bottom()
            # get the facial landmarks
            landmarks = np.array([[p.x, p.y] for p in predictor(frame, rect).parts()])
            # get the left eye landmarks
            left_eye_landmarks = landmarks[LEFT_EYE_POINTS]
            # get the right eye landmarks
            right_eye_landmarks = landmarks[RIGHT_EYE_POINTS]
            # draw contours on the eyes
            left_eye_hull = cv2.convexHull(left_eye_landmarks)
            right_eye_hull = cv2.convexHull(right_eye_landmarks)
            cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
            # compute the EAR for the left eye
            ear_left = eye_aspect_ratio(left_eye_landmarks)
            # compute the EAR for the right eye
            ear_right = eye_aspect_ratio(right_eye_landmarks)
            # compute the average EAR
            EAR_AVG = (ear_left + ear_right) / 2.0
            # detect the eye blink
            if EAR_AVG < EAR_THRESHOLD:
                COUNTER += 1
                if COUNTER >= EAR_CONSEC_FRAMES_CLOSED:
                    #driver asleep: alarm deployed
                    playsound('alarm.wav')
                    
            else:
                if COUNTER >= EAR_CONSEC_FRAMES:
                    BLINKS += 1
                COUNTER = 0

            cv2.putText(frame, "Blinks: {}".format(BLINKS), (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 1)
        cv2.imshow("Driver Drowsiness Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        # When key 'Q' is pressed, exit
        if key is ord('q'):
            break

# release all resources
cap.release()
# destroy all windows
cv2.destroyAllWindows()

