from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model("model_cnn.h5")  # Ensure this file exists after training
labels = ['Floral', 'Geometric', 'Plain', 'PolkaDot', 'Stripe', 'Abstract']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'

    file_path = os.path.join("static", file.filename)
    file.save(file_path)

    img = image.load_img(file_path, target_size=(255, 255))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0) / 255.

    prediction = model.predict(img_tensor, verbose=0)
    pred_class = labels[np.argmax(prediction)]

    return render_template('home.html', prediction=pred_class, image_path=file_path)

if __name__ == '__main__':
    app.run(debug=True)
