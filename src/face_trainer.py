import os
import cv2
import numpy as np
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, '../images/training_images')

face_cascade = cv2.CascadeClassifier('../data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

counter = 0

current_id = 0
label_ids = {}
y_labels = []
x_train = []

print("Training...")

for root, dirs, files, in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("PNG") or file.endswith("jpg") or file.endswith("JPEG"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "_").lower()
            print(label, path)
            if label not in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            # print(label_ids)

            image = cv2.imread(path, 1)

            pil_image = Image.open(path).convert("L")  # convert to grayscale
            image_array = np.array(pil_image, "uint8")
            # print(image_array)

            # size = (550, 550)
            # image_resized = pil_image.resize(size, Image.ANTIALIAS)

            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)
                title = "../images/trained_faces/image" + str(counter) + ".jpg"
                color = (255, 0, 0)
                stroke = 2
                x_coord = x + w
                y_coord = y + h
                cv2.rectangle(image, (x, y), (x_coord, y_coord), color, stroke)
                cv2.imwrite(title, image)
                counter += 1

with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")

print("Complete!")
