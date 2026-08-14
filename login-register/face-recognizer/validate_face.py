from flask import Flask, request, jsonify
import face_recognition
import numpy as np
import cv2
import os
import base64
from io import BytesIO

app = Flask(__name__)

# A function to decode base64 image data to a numpy array (image)
def decode_base64_image(base64_data):
    # Remove the "data:image/jpeg;base64," prefix if it exists
    if base64_data.startswith('data:image/jpeg;base64,'):
        base64_data = base64_data[len('data:image/jpeg;base64,'):]
    
    img_data = base64.b64decode(base64_data)
    image = np.array(bytearray(img_data), dtype=np.uint8)
    return cv2.imdecode(image, cv2.IMREAD_COLOR)

# Function to load the stored face encoding from the uploaded video
def load_stored_face_encoding(user_email):
    # This assumes that the user's face encoding is saved as a .npy file in a folder `face_encodings/`
    encoding_file = f'face_encodings/{user_email}.npy'
    if not os.path.exists(encoding_file):
        return None
    return np.load(encoding_file)

# Endpoint to validate the face
@app.route('/validate_face', methods=['POST'])
def validate_face():
    try:
        # Get the base64 image and email from the POST request
        data = request.get_json()
        image_data = data['image']  # Base64 encoded image
        user_email = data['email']  # Email of the logged-in user
        
        # Decode the base64 image to a format that face_recognition can process
        frame = decode_base64_image(image_data)
        
        # Load the stored face encoding for this user (from the video uploaded at registration)
        stored_face_encoding = load_stored_face_encoding(user_email)
        
        if stored_face_encoding is None:
            return jsonify({'success': False, 'message': 'No face encoding found for this user.'})
        
        # Find faces in the captured frame
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        if len(face_encodings) == 0:
            return jsonify({'success': False, 'message': 'No faces found in the webcam feed.'})
        
        # Compare the captured face with the stored face encoding
        match_results = face_recognition.compare_faces([stored_face_encoding], face_encodings[0])

        # If the face matches, return success
        if match_results[0]:
            return jsonify({'success': True, 'message': 'Face matched successfully!'})
        else:
            return jsonify({'success': False, 'message': 'Face did not match.'})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)  # Run on port 5000
