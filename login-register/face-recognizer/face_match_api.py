from flask import Flask, request, jsonify
import face_recognition
import numpy as np
import base64
import cv2

app = Flask(__name__)

# Load registered face encodings (simulate a database with a file)
def load_registered_encodings():
    try:
        return np.load("registered_encodings.npy", allow_pickle=True).tolist()
    except FileNotFoundError:
        return {}

registered_faces = load_registered_encodings()

@app.route('/register', methods=['POST'])
def register_face():
    """Register a face from an uploaded video."""
    data = request.files
    if 'video' not in data:
        return jsonify({"success": False, "message": "No video uploaded."}), 400

    video_file = data['video']
    video_path = "temp_registration_video.mp4"
    video_file.save(video_path)

    # Extract face encodings from the video
    video_capture = cv2.VideoCapture(video_path)
    face_encodings = []

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        frame_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if frame_encodings:
            face_encodings.append(frame_encodings[0])  # Use the first detected face

    video_capture.release()

    if face_encodings:
        # Save the face encodings (simulate storing in a database)
        user_email = request.form.get("email")
        registered_faces[user_email] = face_encodings
        np.save("registered_encodings.npy", registered_faces)
        return jsonify({"success": True, "message": "Face registered successfully."})
    else:
        return jsonify({"success": False, "message": "No faces detected in the video."}), 400

@app.route('/validate', methods=['POST'])
def validate_face():
    """Validate a face from a webcam frame."""
    data = request.json
    user_email = data.get("email")
    image_data = data.get("image")

    if not user_email or not image_data:
        return jsonify({"success": False, "message": "Invalid input."}), 400

    if user_email not in registered_faces:
        return jsonify({"success": False, "message": "No registration found for this user."}), 404

    # Decode the Base64-encoded image
    image_bytes = base64.b64decode(image_data.split(',')[1])
    nparr = np.frombuffer(image_bytes, np.uint8)
    webcam_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    rgb_frame = cv2.cvtColor(webcam_frame, cv2.COLOR_BGR2RGB)
    webcam_encodings = face_recognition.face_encodings(rgb_frame)

    if not webcam_encodings:
        return jsonify({"success": False, "message": "No face detected in the webcam frame."}), 400

    # Compare webcam face with registered encodings
    matches = face_recognition.compare_faces(registered_faces[user_email], webcam_encodings[0], tolerance=0.5)
    return jsonify({"success": any(matches)})
