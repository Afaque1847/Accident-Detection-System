from flask import Flask, jsonify
import sys
import os
import cv2
from ultralytics import YOLO
import time
import threading


sys.path.append(os.path.join(os.path.dirname(__file__), 'ML_part'))


app = Flask(__name__)


model = YOLO("ML part/best.pt")


stream_url = "/Users/sultanahmad/Downloads/Accident-Detection-and-Notification-main/ML part/inputs/videos/video1.mp4"


frame_width = 640
frame_height = 480


class_id = -1
confidence_score = 0.0


cap = cv2.VideoCapture(stream_url)
frame_skip = 5  
frame_count = 0
fps = 10
time_delay = 1 / fps  


def perform_detection(frame):
    global class_id, confidence_score
    resized_frame = cv2.resize(frame, (frame_width, frame_height))

   
    results = model(resized_frame)

    
    if results[0].boxes and len(results[0].boxes.cls) > 0:
        
        class_id = int(results[0].boxes.cls[0].item())
        confidence_score = float(results[0].boxes.conf[0].item())
        
        
        box = results[0].boxes.xyxy[0]  
        x1, y1, x2, y2 = map(int, box)
        label = f'Class {class_id}: {confidence_score:.2f}'
        cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(resized_frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        
        class_id = -1
        confidence_score = 0.0

    return resized_frame  


@app.route('/predict', methods=['GET'])
def predict():
    global class_id, confidence_score

    
    is_frame, frame = cap.read()
    if not is_frame:
        return jsonify({"error": "Video stream ended"}), 400

    
    perform_detection(frame)

    
    return jsonify({
        'class_id': class_id,
        'confidence': confidence_score
    })


def process_video():
    global frame_count
    try:
        while True:
            is_frame, frame = cap.read()
            if not is_frame:
                break

            if frame_count % frame_skip == 0:
               
                frame_with_boxes = perform_detection(frame)
            else:
                frame_with_boxes = cv2.resize(frame, (frame_width, frame_height))

            
            cv2.imshow('Accident Detection Video', frame_with_boxes)

            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(time_delay)
            frame_count += 1
    except KeyboardInterrupt:
        print("Program interrupted.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)).start()

    
    process_video()



# from flask import Flask, jsonify
# import sys
# import os
# import cv2
# from ultralytics import YOLO
# import time
# import threading

# # Add the ML_part directory to sys.path
# sys.path.append(os.path.join(os.path.dirname(__file__), 'ML_part'))

# # Initialize Flask app
# app = Flask(__name__)

# # Initialize YOLO model
# model = YOLO("ML part/best.pt")

# # Define the path for the video
# stream_url = "/Users/sultanahmad/Downloads/Accident-Detection-and-Notification-main/ML part/inputs/videos/video4.mp4"

# # Set frame dimensions for resizing
# frame_width = 640
# frame_height = 480

# # Global variable to store predictions
# class_id = -1
# confidence_score = 0.0

# # Video capture initialization
# cap = cv2.VideoCapture(stream_url)
# frame_skip = 5  # Process every 5th frame
# frame_count = 0
# fps = 10
# time_delay = 1 / fps  # Time delay based on FPS

# # Function for object detection
# def perform_detection(frame):
#     global class_id, confidence_score
#     resized_frame = cv2.resize(frame, (frame_width, frame_height))

#     # Run YOLO directly on the resized frame
#     results = model(resized_frame)

#     # Check if there are any detections
#     if results[0].boxes and len(results[0].boxes.cls) > 0:
#         # Extract the class ID and confidence score
#         class_id = int(results[0].boxes.cls[0].item())
#         confidence_score = float(results[0].boxes.conf[0].item())
#     else:
#         # No detections
#         class_id = -1
#         confidence_score = 0.0

#     return class_id, confidence_score

# # Flask route to get predictions
# @app.route('/predict', methods=['GET'])
# def predict():
#     global class_id, confidence_score

#     # Read the next frame from the video stream
#     is_frame, frame = cap.read()
#     if not is_frame:
#         return jsonify({"error": "Video stream ended"}), 400
    
#     # Perform object detection
#     perform_detection(frame)
    
#     # Return the prediction as a JSON response
#     return jsonify({
#         'class_id': class_id,
#         'confidence': confidence_score
#     })

# # Function to process the video stream
# def process_video():
#     global frame_count
#     try:
#         while True:
#             is_frame, frame = cap.read()
#             if not is_frame:
#                 break
#             if frame_count % frame_skip == 0:
#                 # Perform detection on the frame
#                 perform_detection(frame)
#                 time.sleep(time_delay)
#             frame_count += 1
#     except KeyboardInterrupt:
#         print("Program interrupted.")
#     finally:
#         # Clean up the video capture
#         cap.release()
#         cv2.destroyAllWindows()

# if __name__ == '__main__':
#     # Start the Flask app in a separate thread
#     threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)).start()

#     # Start processing the video stream
#     process_video()



# from flask import Flask, jsonify
# import sys
# import os
# import cv2
# from ultralytics import YOLO
# import time

# # Add the ML_part directory to sys.path
# sys.path.append(os.path.join(os.path.dirname(__file__), 'ML_part'))

# # Now import the Detection class from detection module
# from detection import Detection

# # Initialize Flask app
# app = Flask(__name__)

# # Initialize YOLO model
# model = YOLO("ML part/best.pt")

# # Define the path for the video
# stream_url = "/Users/sultanahmad/Downloads/Accident-Detection-and-Notification-main/ML part/inputs/videos/video4.mp4"

# # Set frame dimensions for resizing
# frame_width = 640
# frame_height = 480

# # Global variable to store predictions
# class_id = -1
# confidence_score = 0.0

# # Video capture initialization
# cap = cv2.VideoCapture(stream_url)
# frame_skip = 5  # Process every 5th frame
# frame_count = 0
# fps = 10
# time_delay = 1 / fps  # Time delay based on FPS

# # Function for object detection
# def perform_detection(frame):
#     global class_id, confidence_score
#     resized_frame = cv2.resize(frame, (frame_width, frame_height))

#     # Save the resized frame to a temporary path
#     temp_image_path = "ML part/temp/temp.jpg"
#     cv2.imwrite(temp_image_path, resized_frame)

#     # Perform prediction on the frame
#     results = model.predict(source=temp_image_path)
    
#     # Assuming the model returns the class ID and confidence score
#     class_id = results[0].boxes.cls[0].item()  # Accessing the class ID of the first detected object
#     confidence_score = results[0].boxes.conf[0].item()  # Confidence of the detection

#     return class_id, confidence_score

# # Flask route to get predictions
# @app.route('/predict', methods=['GET'])
# def predict():
#     global class_id, confidence_score
    
#     # Read the next frame from the video stream
#     is_frame, frame = cap.read()
#     if not is_frame:
#         return jsonify({"error": "Video stream ended"}), 400
    
#     # Perform object detection
#     perform_detection(frame)
    
#     # Return the prediction as a JSON response
#     return jsonify({
#         'class_id': class_id,
#         'confidence': confidence_score
#     })

# if __name__ == '__main__':
#     # Start the Flask app in a separate thread to run alongside video processing
#     import threading
#     threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000)).start()

#     # Process frames for video stream (this will be done while Flask server is running)
#     try:
#         while True:
#             is_frame, frame = cap.read()
#             if not is_frame:
#                 break
#             if frame_count % frame_skip == 0:
#                 # Perform detection on the frame and process
#                 perform_detection(frame)
#                 time.sleep(time_delay)
#             frame_count += 1
#     except KeyboardInterrupt:
#         print("Program interrupted.")
#     finally:
#         # Clean up the video capture and close all windows
#         cap.release()
#         cv2.destroyAllWindows()



# from ultralytics import YOLO
# import cv2

# # Load a model
# model = YOLO("ML part/best.pt")

# # Set the dimensions for captured frames
# frame_width = 640
# frame_height = 480

# # known freely available urls with traffic stream:
# # http://204.106.237.68:88/mjpg/1/video.mjpg

# # URL of the video stream - preferably use motion jpg urls
# stream_url = "/Users/sultanahmad/Downloads/Accident-Detection-and-Notification-main/ML part/inputs/videos/video4.mp4"

# # start capturing
# cap = cv2.VideoCapture(stream_url)

# while cap.isOpened():
#     # boolean success flag and the video frame
#     # if the video has ended, the success flag is False
#     is_frame, frame = cap.read()
#     if not is_frame:
#         break

#     resized_frame = cv2.resize(frame, (frame_width, frame_height))

#     # Save the resized frame as an image in a temporary directory
#     temp_image_path = "ML part/temp/temp.jpg"
#     cv2.imwrite(temp_image_path, resized_frame)

#     # Perform object detection on the image and show the results
#     model.predict(source=temp_image_path, show=True)

#     # Check for the 'q' key press to quit
#     if cv2.waitKey(1) == 0xff & ord('q'):
#         break

# # Release the video capture and close any OpenCV windows
# cap.release()
# cv2.destroyAllWindows()

