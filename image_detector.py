import cv2

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the image you want to test
# Make sure the image is in the same directory as your script
img = cv2.imread('test_image.png')

# Convert the image to grayscale (Haar cascades work best on grayscale images)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Perform face detection
# detectMultiScale(image, scaleFactor, minNeighbors)
# scaleFactor: How much the image size is reduced at each image scale.
# minNeighbors: How many neighbors each candidate rectangle should have to retain it.
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# Draw a rectangle around the detected faces
for (x, y, w, h) in faces:
    # cv2.rectangle(image, start_point, end_point, color, thickness)
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Display the output
cv2.imshow('Detected Faces', img)

# Wait until a key is pressed to close the window
cv2.waitKey(0)
cv2.destroyAllWindows()