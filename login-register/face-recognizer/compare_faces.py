import face_recognition
import numpy as np
import sys
import os

# Get the file path and user email from the command line arguments
webcam_image_path = sys.argv[1]  # Path to the webcam image
user_email = sys.argv[2]  # User's email

# Load the saved face encoding for the user
encoding_file = f'face_encoding/face_encodings/{user_email}.npy'

if not os.path.exists(encoding_file):
    print(f"Error: No face encoding found for {user_email}")
    sys.exit(1)

saved_encoding = np.load(encoding_file)

# Load the webcam image and get its face encoding
webcam_image = face_recognition.load_image_file(webcam_image_path)
webcam_encoding = face_recognition.face_encodings(webcam_image)

if len(webcam_encoding) == 0:
    print("No face detected in the webcam image.")
    sys.exit(1)

# Compare the webcam face encoding with the saved face encoding
results = face_recognition.compare_faces([saved_encoding], webcam_encoding[0], tolerance=0.5)  # Adjust tolerance here

if results[0]:
    print("Face match successful!")
else:
    print("Face does not match.")
