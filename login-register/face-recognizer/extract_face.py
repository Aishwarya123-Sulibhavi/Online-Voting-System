import face_recognition
import cv2
import numpy as np
import os
import sys

# Get the video path and email from the arguments passed from PHP
video_file = sys.argv[1]  # First argument is the video path
user_email = sys.argv[2]  # Second argument is the user's email

# Function to extract face encoding from a video file
def extract_face_encoding_from_video(video_file, user_email):
    # Open the video file
    video_capture = cv2.VideoCapture(video_file)

    if not video_capture.isOpened():
        print(f"Error: Unable to open video file {video_file}")
        return

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Find all face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        # If a face is found, save the encoding (you can choose to use the first face detected)
        if face_encodings:
            face_encoding = face_encodings[0]
            encoding_file = f'face_encodings/{user_email}.npy'

            # Make sure the directory exists
            if not os.path.exists('face_encodings'):
                os.makedirs('face_encodings')

            # Save the face encoding
            np.save(encoding_file, face_encoding)
            print(f"Face encoding saved for {user_email}.")
            break  # Save only the first face encoding found in the video

    # Release the video capture object
    video_capture.release()

# Call the function to process the video and extract the face encoding
extract_face_encoding_from_video(video_file, user_email)
