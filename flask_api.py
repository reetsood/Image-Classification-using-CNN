import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import requests

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "flower_classification_model.keras"
model = load_model(MODEL_PATH)

# Define class labels
class_labels = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

@app.route('/')
def home():
    return render_template("frontend.html")

def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(180,180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])  # Apply softmax

    predicted_class = class_labels[np.argmax(result)]
    confidence_score = np.max(result) * 100

    outcome = {
        "class": predicted_class,
        "confidence": confidence_score
    }
    return outcome

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'GET':
        return "Use POST with an image file to get predictions.", 405

    file = request.files['file']
    filepath = "temp_image.jpg"
    file.save(filepath)

    result = classify_images(filepath)

    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):  # Convert NumPy array to list
            return obj.tolist()
        elif isinstance(obj, np.float32) or isinstance(obj, np.float64):  # Convert float32/64 to Python float
            return float(obj)
        elif isinstance(obj, dict):  # If result is a dictionary, convert all values
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):  # If result is a list, convert all elements
            return [convert_to_serializable(value) for value in obj]
        else:
            return obj  # Return as-is if it's already serializable

    # Convert the result to a serializable format
    serializable_result = convert_to_serializable(result)

    return jsonify(serializable_result)


if __name__ == '__main__':
    app.run(debug=True)

