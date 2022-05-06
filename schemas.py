from pydantic import BaseModel
from typing import Optional,List

class Data(BaseModel):
    image: str
    similarity:int

class FaceDetect(BaseModel):
    face_detect: bool
    no_of_person: int
    landmark_image: Optional[str] = None
    match: Optional[List[Data]] = None
  
