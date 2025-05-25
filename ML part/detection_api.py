# from flask import Flask, jsonify
# import sys
# import os

# # Add the ML_part directory to sys.path
# sys.path.append(os.path.join(os.path.dirname(__file__), 'ML_part'))

# # Now import Detection class
# from detection import Detection

# app = Flask(__name__)

# @app.route('/predict', methods=['GET'])
# def predict():
#     class_id, confidence = Detection.static_detection()  # 🔥 corrected method name
#     return jsonify({
#         'class_id': class_id,
#         'confidence': confidence
#     })

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)

