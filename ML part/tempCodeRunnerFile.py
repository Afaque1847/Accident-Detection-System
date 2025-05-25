
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

