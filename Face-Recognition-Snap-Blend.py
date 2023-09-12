import os
import cv2
import numpy as np

# Load the pre-trained Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create a dictionary to store known face encodings and their corresponding names
known_faces = {}
known_faces_dir = 'C:/Users/Dell/Desktop/__pycache__/preview/images/'

for person_folder in os.listdir(known_faces_dir):
    person_name = person_folder.split('.')[0]
    person_images_dir = os.path.join(known_faces_dir, person_folder)
    person_images = [os.path.join(person_images_dir, image) for image in os.listdir(person_images_dir)]

    # Create an empty list to store the face encodings for this person
    person_face_encodings = []

    for image_path in person_images:
        img = cv2.imread(image_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Assuming each image contains only one face, use the first detected face for recognition
        if len(faces) == 1:
            x, y, w, h = faces[0]
            face_encoding = gray_img[y:y + h, x:x + w]

            # Resize face_encoding to a fixed size for consistent comparisons
            target_size = (100, 100)
            face_encoding = cv2.resize(face_encoding, target_size)

            person_face_encodings.append(face_encoding)

    # Store the face encodings for this person in the known_faces dictionary
    known_faces[person_name] = person_face_encodings

# Start capturing video from the default camera (0)
video_capture = cv2.VideoCapture(0)

while True:
    # Capture each frame from the video feed
    ret, frame = video_capture.read()

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the current frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop through each detected face
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Crop the face region for recognition
        face_region = gray_frame[y:y + h, x:x + w]

        # Resize face_region to a fixed size for consistent comparisons
        target_size = (100, 100)
        face_region = cv2.resize(face_region, target_size)

        # Compare the face with known faces using cv2.matchTemplate
        found_name = "Unknown"
        for name, known_encodings in known_faces.items():
            for known_encoding in known_encodings:
                result = cv2.matchTemplate(face_region, known_encoding, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)

                # If there's a match, set the person's name
                if max_val > 0.9:  # You can adjust this threshold to control the recognition sensitivity
                    found_name = name
                    break

        # Display the name
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, found_name, (x + 6, y + h - 6), font, 0.5, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close the window
video_capture.release()
cv2.destroyAllWindows()
