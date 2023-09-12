import cv2
import threading

# Function to process a frame
def process_frame(frame):
    # Perform some processing on the frame
    processed_frame = frame  # Placeholder, replace with actual processing code

    # Display the processed frame
    cv2.imshow('Processed Frame', processed_frame)

# Initialize the face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Flag to indicate when to stop
stop_flag = False

# Function to continuously read and process frames
def process_frames():
    while not stop_flag:
        # Read the frame from the webcam
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, 1.2, 6)

        # Process each detected face
        for (x, y, w, h) in faces:
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Start a new thread to process the frame asynchronously
        thread = threading.Thread(target=process_frame, args=(frame,))
        thread.start()

        # Display the frame
        cv2.imshow('Face Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Start the frame processing thread
process_thread = threading.Thread(target=process_frames)
process_thread.start()

# Wait for the process_thread to finish
process_thread.join()
