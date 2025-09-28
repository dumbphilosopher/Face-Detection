import cv2
from mtcnn.mtcnn import MTCNN

# Initialize the MTCNN detector
detector = MTCNN()

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB for MTCNN
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    faces = detector.detect_faces(rgb_frame)
    
    # Draw rectangles and keypoints
    for face in faces:
        x, y, w, h = face['box']
        keypoints = face['keypoints']
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.circle(frame, (keypoints['left_eye']), 2, (0,0,255), 2)
        cv2.circle(frame, (keypoints['right_eye']), 2, (0,0,255), 2)
        cv2.circle(frame, (keypoints['nose']), 2, (0,0,255), 2)
        cv2.circle(frame, (keypoints['mouth_left']), 2, (0,0,255), 2)
        cv2.circle(frame, (keypoints['mouth_right']), 2, (0,0,255), 2)

    # Display the resulting frame
    cv2.imshow('Real-time MTCNN', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()