import cv2
# The main class from the library is called MTCNN
from mtcnn.mtcnn import MTCNN

# Load the image
img = cv2.imread('test_image.png')
# OpenCV loads images in BGR format. MTCNN expects RGB. We need to convert it.
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Initialize the MTCNN detector
detector = MTCNN()

# The detect_faces method returns a list of dictionaries
faces = detector.detect_faces(rgb_img)

# Process the results
for face in faces:
    # Get the bounding box coordinates
    x, y, width, height = face['box']
    
    # Get the keypoints (landmarks)
    keypoints = face['keypoints']
    
    # Draw the bounding box
    cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 2)
    
    # Draw circles on the keypoints
    cv2.circle(img, (keypoints['left_eye']), radius=2, color=(0, 0, 255), thickness=-1)
    cv2.circle(img, (keypoints['right_eye']), radius=2, color=(0, 0, 255), thickness=-1)
    cv2.circle(img, (keypoints['nose']), radius=2, color=(0, 0, 255), thickness=-1)
    cv2.circle(img, (keypoints['mouth_left']), radius=2, color=(0, 0, 255), thickness=-1)
    cv2.circle(img, (keypoints['mouth_right']), radius=2, color=(0, 0, 255), thickness=-1)

# Display the output
cv2.imshow('MTCNN Face Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()