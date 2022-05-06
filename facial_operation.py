import cv2
import numpy as np 
import face_recognition 
import pickle
import mediapipe as mp
from PIL import Image
import base64
import math
import shutil

images = []
classNames = []



def img_to_base64(imageFullPath):
    with open(imageFullPath, "rb") as img_file:
        encoded_image_string = base64.b64encode(img_file.read())
    return (encoded_image_string)


def save_image(image_path, image):
    print(image_path)
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(image, buffer)

def find_encoding(imagePath):
    imgFile = cv2.imread(imagePath)
    Img = cv2.cvtColor(imgFile, cv2.COLOR_BGR2RGB)
    return face_recognition.face_encodings(Img)

def find_landmarks(img):
    imgFile = cv2.imread(img)
    mp_face_mesh = mp.solutions.face_mesh
    # Load drawing_utils and drawing_styles
    mp_drawing = mp.solutions.drawing_utils 
    mp_drawing_styles = mp.solutions.drawing_styles         
    # Run MediaPipe Face Mesh.
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        refine_landmarks=True,
        max_num_faces=2,
        min_detection_confidence=0.5) as face_mesh:
        # Convert the BGR image to RGB and process it with MediaPipe Face Mesh.
        results = face_mesh.process(cv2.cvtColor(imgFile, cv2.COLOR_BGR2RGB))
        
        face_landmarks =  results.multi_face_landmarks[0]
        mp_drawing.draw_landmarks(
        image=imgFile,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
        image=imgFile,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(
        image=imgFile,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_IRISES,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_iris_connections_style())
        imgFile= cv2.cvtColor(imgFile,cv2.COLOR_BGR2RGB)
        pil_image=Image.fromarray(imgFile)
    return(pil_image)

def findMatches(imagePath):
    with open(".\dataset_faces.dat", 'rb') as f:
        encodeListKnown = pickle.load(f)
    classNames = list(encodeListKnown.keys())
    unknownEncoding=find_encoding(imagePath)
    faceDist = face_recognition.face_distance(list(encodeListKnown.values()),unknownEncoding[0])
    #get index of top 4 smallest values
    minIndexes = np.argpartition(faceDist,4)[:4]
    minClasses = [classNames[i] for i in minIndexes]
    minDist = faceDist[np.argpartition(faceDist,4)[:4]]
    minSim = 1/np.exp(minDist)
    minSim=np.where(minSim>0.6,minSim,minSim-1/math.exp(1))
    minSim = minSim *100
    return(minClasses, minSim)