from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
import os
import numpy as np
from PIL import Image
from flask import send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the trained model
model = load_model('model.h5', compile=False)  # Change 'model.h5' to the filename of your trained model

# Define classes
classes = ['INFECTED WITH COVID', 'COMPLETELY NORMAL', 'DIAGNOSED WITH NON COVID PNEUMONIA']


# Define GIFs for each class
gifs = {
    'COVID': 'covid.gif',
    'Normal': 'normal.gif',
    'Pneumonia': 'pneumonia.gif'
}

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for image upload and classification
@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        # Save the uploaded file
        file_path = os.path.join(app.root_path, 'uploads', secure_filename(file.filename))
        print("Saving file to:", file_path)
        file.save(file_path)
        # Perform classification
        result = classify_image(file_path)
        # Redirect to a new page with the result
        return redirect(url_for('result_page', result=result, filename=file.filename))
    else:
        return redirect(url_for('home'))  # Redirect to home page if not a POST request

# Function to classify an image
def classify_image(file_path):
    # Load and preprocess the image
    img = Image.open(file_path)
    img = img.resize((224, 224))  # Resize image to match input shape of the model
    img_rgb = Image.new("RGB", img.size)  # Create a new RGB image
    img_rgb.paste(img)  # Paste the grayscale image onto the RGB image
    img = np.array(img_rgb) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension  # Add batch dimension

    # Perform prediction
    predictions = model.predict(img)
    predicted_class_index = np.argmax(predictions)
    predicted_class = classes[predicted_class_index]

    # Delete the uploaded file

    return predicted_class


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    uploads_dir = os.path.join(app.root_path, 'uploads')
    return send_from_directory(uploads_dir, filename)

# Define route for displaying the result on a separate page
@app.route('/result/<result>/<filename>')
def result_page(result,filename):
    gif = gifs.get(result, 'default.gif')  # Get the corresponding GIF for the result
    return render_template('result.html', result=result, gif=gif, filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
