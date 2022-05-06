import encodings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from fastapi import  File, Form, HTTPException, Depends, status
from fastapi.datastructures import UploadFile
import glob
import schemas,facial_operation

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REF_IMAGES = '.\images\\' 
INPUT_IMAGES= ".\inputImage\\"
LANDMARK_IMAGES = ".\landmarkImage\\"

@ app.post("/search", response_model=schemas.FaceDetect, status_code=status.HTTP_200_OK)
async def face_detect(image: UploadFile=File(...), facial_landmarks: Optional[bool]=True):
    result={}
    facial_operation.save_image(INPUT_IMAGES+image.filename,image.file)
    encoding = facial_operation.find_encoding(f"{INPUT_IMAGES}{image.filename}")
    faceCount = len(encoding)
    detection = faceCount != 0
    result={"face_detect": detection, "no_of_person": faceCount}
    if (faceCount ==1):
        imgLandmarks=facial_operation.find_landmarks(f"{INPUT_IMAGES}{image.filename}")
        imgLandmarks.save(LANDMARK_IMAGES + image.filename)
        if facial_landmarks:
            encodedLandmarkImage=facial_operation.img_to_base64(LANDMARK_IMAGES + image.filename)
            result["landmark_image"] = encodedLandmarkImage
        classes, similarites = facial_operation.findMatches(f"{INPUT_IMAGES}{image.filename}")
        match =[]
        for idx, label in enumerate(classes):
            imagePath =  glob.glob((f"{REF_IMAGES}\{label}.*"))
            if not imagePath:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Reference Image not found")
            if (len(imagePath)) > 1:
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Multiple Reference Image found")
            strImage=facial_operation.img_to_base64(imagePath[0])
            match.append({"image":strImage,"similarity":similarites[idx]})
        match = sorted(match,key= lambda x:x["similarity"], reverse=True)
        result["match"] = match
    return result


    
    