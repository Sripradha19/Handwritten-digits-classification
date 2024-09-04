import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from PIL import Image
import io

app = Flask(__name__)

# Load the model from the .h5 file
model = load_model(r'C:\Users\rsrip\Desktop\CG_Project\CG_Project\digits.h5')

def preprocess_image(image):
    # Convert to grayscale
    img = image.convert('L')
    img = img.resize((28, 28))  # Resize to 28x28 pixels

    # Convert image to numpy array
    img_array = np.array(img)

    # Normalize the image
    img_array = img_array.astype('float32') / 255

    # Reshape to fit the model input
    img_array = img_array.reshape(1, 28, 28, 1)
    
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No selected file')
    if file:
        image = Image.open(io.BytesIO(file.read()))
        image = preprocess_image(image)

        # Make prediction
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)[0]

        return render_template('index.html', predicted_class=int(predicted_class))

if __name__ == '__main__':
    app.run(debug=True)
