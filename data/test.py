from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch

def speak(str1):
    speaker = Dispatch("SAPI.SpVoice")
    speaker.Speak(str1)

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Load data
with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

print('Shape of Faces matrix --> ', FACES.shape)
print('Length of labels -->', len(LABELS))

# Ensure labels and faces match in length
if len(LABELS) != len(FACES):
    min_len = min(len(LABELS), len(FACES))
    LABELS = LABELS[:min_len]
    FACES = FACES[:min_len]
    print(f"[INFO] Trimmed to: {len(FACES)} samples")

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

imgBackground = cv2.imread("background.png")
if imgBackground is None:
    imgBackground = np.zeros((720, 1280, 3), dtype=np.uint8)

COL_NAMES = ['NAME', 'TIME']

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY), (45, 40)).flatten().reshape(1, -1)

        output = knn.predict(resized_img)
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        exist = os.path.isfile("Attendance/Attendance_" + date + ".csv")

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, str(output[0]), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        attendance = [str(output[0]), str(timestamp)]

    imgBackground[162:162 + 480, 55:55 + 640] = frame
    cv2.imshow("Frame", imgBackground)
    k = cv2.waitKey(1)

    if k == ord('o'):
        speak("Attendance has been Taken successfully.")
        time.sleep(2)
        with open("Attendance/Attendance_" + date + ".csv", "a", newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not exist:
                writer.writerow(COL_NAMES)
            writer.writerow(attendance)

    if k == ord('p'):
        break

video.release()
cv2.destroyAllWindows()
