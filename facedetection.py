# ProjectGurukul Face Counter

# Import necessary packages
import cv2
import face_recongition 
import mediapipe as mp

# Define mediapipe Face detector
face_detection = mp.solutions.face_detection.FaceDetection(0.6)

# Detection function
def detector(frame):
    count = 0
    height, width, channel = frame.shape

    # Convert frame BGR to RGB colorspace
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect results from the frame
    result = face_detection.process(imgRGB)

    # Extract information from the result
    try:
        for count, detection in enumerate(result.detections):
            # Extract score and bounding box information
            score = detection.score
            box = detection.location_data.relative_bounding_box

            x, y, w, h = int(box.xmin * width), int(box.ymin * height), int(box.width * width), int(box.height * height)
            score = str(round(score[0] * 100, 2))

            # Draw rectangles and labels
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y - 25), (0, 0, 255), -1)
            cv2.putText(frame, score, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        count += 1
        print("Found", count, "Faces!")
        
    except:
        pass

    return count, frame

# Use Webcam for Face Detection
cap = cv2.VideoCapture(0)  # Webcam index (0 for default webcam)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from webcam.")
        break

    count, output = detector(frame)
    cv2.putText(output, "Number of Faces: " + str(count), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Webcam - Face Counter", output)

    # Exit on pressing 'q'
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
