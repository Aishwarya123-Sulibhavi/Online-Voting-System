import sys
import json
import face_recognition

def main():
    if len(sys.argv) != 3:
        print(json.dumps({"match": False, "error": "Invalid arguments"}))
        sys.exit(1)

    registered_image_path = sys.argv[1]
    temp_image_path = sys.argv[2]

    try:
        # Load the registered image
        registered_image = face_recognition.load_image_file(registered_image_path)
        registered_encodings = face_recognition.face_encodings(registered_image)
        if len(registered_encodings) == 0:
            print(json.dumps({"match": False, "error": "No face found in registered image"}))
            sys.exit(1)

        # Load the temporary image
        temp_image = face_recognition.load_image_file(temp_image_path)
        temp_encodings = face_recognition.face_encodings(temp_image)
        if len(temp_encodings) == 0:
            print(json.dumps({"match": False, "error": "No face found in uploaded frame"}))
            sys.exit(1)

        # Compare the faces
        results = face_recognition.compare_faces([registered_encodings[0]], temp_encodings[0])

        if results[0]:
            print(json.dumps({"match": True}))
        else:
            print(json.dumps({"match": False}))
    except Exception as e:
        print(json.dumps({"match": False, "error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
