from flask import Flask, request, jsonify
import numpy as np
import cv2
from keras.models import load_model
import io

app = Flask(__name__)

# Load the trained model
model = load_model('mask_detector.model.h5')


@app.route('/predict', methods=['POST'])
def predict():
    # Check if the request contains an image
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    # Load and preprocess the image
    image_file = request.files['image']
    image_bytes = io.BytesIO(image_file.read())
    image = cv2.imdecode(np.frombuffer(image_bytes.read(), np.uint8), -1)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    # Make prediction using the loaded model
    prediction = model.predict(image)
    prediction_label = "Mask" if prediction[0][0] > 0.5 else "No Mask"

    return jsonify({'prediction': prediction_label})


if __name__ == '__main__':
    app.run(debug=True)
