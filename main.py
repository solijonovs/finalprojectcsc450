from flask import Flask, render_template, Response, request, jsonify
import cv2
import face_recognition

app = Flask(__name__)

# Load a sample picture and learn how to recognize it.
known_image = face_recognition.load_image_file("path_to_known_image.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

# Initialize the camera
video_capture = cv2.VideoCapture(0)

def generate_video_stream():
    """Generate a video stream for live face recognition."""
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if not ret:
            break

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Detect faces
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        # Check for known faces
        for face_encoding in face_encodings:
            match = face_recognition.compare_faces([known_encoding], face_encoding)
            if match[0]:
                cv2.putText(frame, "Authenticated", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Not Recognized", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Encode frame to JPEG
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

# Release the video capture when done
video_capture.release()
cv2.destroyAllWindows()
