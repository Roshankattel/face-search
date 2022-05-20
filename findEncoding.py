import cv2 
import face_recognition 
import os
import pickle

images = []
classNames = []

imagePath = '.\images'
# imagePath = "./testImage"
List = os.listdir(imagePath)

for cl in List:
    curImg = cv2.imread(f'{imagePath}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

# print(classNames)

def findEncodings(imgaes):
    encodeList ={}
    for count, img in enumerate(images):
        print(classNames[count])
        try:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #coverting images to RGB
            encodeList[classNames[count]] = face_recognition.face_encodings(img)[0]
        except :
            continue
        # print(count)
    return(encodeList)


encodeListKnown = findEncodings(images)
print ('Encoding Complete')

#saving the face encodings
with open('dataset_faces_500.dat', 'wb') as f:
    pickle.dump(encodeListKnown, f)