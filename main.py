from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
import cv2
import numpy as np
import base64
from pydantic import BaseModel
from sqlalchemy.orm import Session
from models import User, Attendance
from database import get_db
from auth import verify_password, get_password_hash, create_access_token, get_current_user
from datetime import timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174"],
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

class ImageRequest(BaseModel):
    image: str

class UserCreate(BaseModel):
    username: str
    password: str
    reference_image: str

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None, None
    (x, y, w, h) = faces[0]
    face_region = gray[y:y+h, x:x+w]
    face_region = cv2.resize(face_region, (100, 100), interpolation=cv2.INTER_AREA)
    return face_region, (x, y, w, h)

def compare_faces(ref_face, detected_face):
    if ref_face.shape != detected_face.shape:
        detected_face = cv2.resize(detected_face, ref_face.shape[::-1], interpolation=cv2.INTER_AREA)
    
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(ref_face, None)
    keypoints2, descriptors2 = orb.detectAndCompute(detected_face, None)
    
    if descriptors1 is None or descriptors2 is None:
        logger.info("No keypoints detected in one or both images")
        return False
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    good_matches = [m for m in matches if m.distance < 50]
    logger.info(f"Good matches found: {len(good_matches)}")
    
    return len(good_matches) > 5  # Adjusted threshold

@app.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    access_token = create_access_token(data={"sub": user.username}, expires_delta=timedelta(minutes=30))
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/register")
async def register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    img_data = base64.b64decode(user.reference_image)
    nparray = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparray, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Failed to decode reference image")
    
    face_region, box = extract_face(img)
    if face_region is None:
        raise HTTPException(status_code=400, detail="No face detected in reference image")
    
    reference_face_data = base64.b64encode(face_region.tobytes()).decode('utf-8')
    
    hashed_password = get_password_hash(user.password)
    new_user = User(username=user.username, hashed_password=hashed_password, reference_face=reference_face_data)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "User created"}

@app.post("/recognize")
async def detect_and_mark_attendance(
    request: ImageRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        logger.info("Decoding image")
        img_data = base64.b64decode(request.image)
        nparray = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparray, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")
        
        logger.info("Detecting face")
        detected_face, detected_box = extract_face(img)
        if detected_face is None:
            raise HTTPException(status_code=400, detail="No faces detected")
        
        detected_faces = [{
            "x": int(detected_box[0]), 
            "y": int(detected_box[1]), 
            "width": int(detected_box[2]), 
            "height": int(detected_box[3])
        }]

        logger.info("Retrieving reference face")
        if not current_user.reference_face:
            raise HTTPException(status_code=400, detail="No reference face data for this user")
        
        ref_face = np.frombuffer(base64.b64decode(current_user.reference_face), dtype=np.uint8)
        ref_face = ref_face.reshape(100, 100)

        logger.info("Comparing faces")
        match = compare_faces(ref_face, detected_face)
        if match:
            logger.info("Face matched, marking attendance")
            attendance = Attendance(
                user_id=current_user.id,
                face_x=detected_faces[0]["x"],
                face_y=detected_faces[0]["y"],
                face_width=detected_faces[0]["width"],
                face_height=detected_faces[0]["height"]
            )
            db.add(attendance)
            db.commit()
            return {"faces": detected_faces, "message": f"Attendance marked for {current_user.username}"}
        else:
            logger.info("Face did not match")
            raise HTTPException(status_code=403, detail="Detected face does not match your reference face")
    except HTTPException as he:
        logger.error(f"HTTPException: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/attendance")
async def get_attendance(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    records = db.query(Attendance).filter(Attendance.user_id == current_user.id).all()
    return [{"id": r.id, "timestamp": r.timestamp, "x": r.face_x, "y": r.face_y, "width": r.face_width, "height": r.face_height} 
            for r in records]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)