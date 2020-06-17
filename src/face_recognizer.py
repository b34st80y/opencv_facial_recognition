import cv2
import pickle

face_cascade = cv2.CascadeClassifier('../data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {}
with open("labels.pickle", 'rb') as f:
    labels = pickle.load(f)
    labels = {v: k for k, v in labels.items()}

cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
print("Running...")
while True:
    ret, frame = cap.read()

    grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayImg)

    for (x, y, w, h) in faces:
        roi_gray = grayImg[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Recognize
        id_, conf = recognizer.predict(roi_gray)
        # print(conf)
        if conf <= 85:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 0, 0)
            stroke = 2
            text = str(round(conf)) + ": " + name
            cv2.putText(frame, text, (x, y), font, 1, color, stroke, cv2.LINE_AA)

        color = (255, 0, 0)
        stroke = 2
        x_coord = x + w
        y_coord = y + h
        cv2.rectangle(frame, (x, y), (x_coord, y_coord), color, stroke)

    cv2.imshow('image', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

print("Stopped!")
cap.release()
cv2.destroyAllWindows()
