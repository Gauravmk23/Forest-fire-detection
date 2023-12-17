from flask import Flask, render_template, request, redirect, url_for, send_file
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import os
from tqdm import tqdm

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('static/ForestFire_c_Model.h5')

def classify_image(image_path):
    img = image.load_img(image_path, target_size=(256, 256))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    predictions = model.predict(img, verbose=0)

    probability_fire = predictions[0][0] * 100
    probability_normal = predictions[0][1] * 100

    if probability_normal > 50:
        classification = "normal"
    else:
        classification = "fire"

    return classification, probability_fire, probability_normal

@app.route('/index', methods=['GET', 'POST'])
def index():
    classification = None
    probability_fire = None
    probability_normal = None

    if request.method == 'POST':
        uploaded_file = request.files['image']
        if uploaded_file.filename != '':
            image_path = os.path.join('uploads', uploaded_file.filename)
            uploaded_file.save(image_path)

            classification, probability_fire, probability_normal = classify_image(image_path)

    return render_template('index.html', classification=classification, probability_fire=probability_fire, probability_normal=probability_normal)

@app.route('/')
def main():
    return render_template('main.html')

@app.route('/video')
def upload():
    return render_template('upload.html')

@app.route('/process_video', methods=['POST'])
def process_video():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    # Define video paths
    input_video_path = 'input_video.mp4'
    output_video_path = 'output_video.avi'

    file.save(input_video_path)

    # Open the video
    cap = cv2.VideoCapture(input_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = tqdm(total=total_frames, position=0, leave=True, desc="Processing Frames")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_resized = cv2.resize(frame, (256, 256))
        frame_resized = image.img_to_array(frame_resized)
        frame_resized = np.expand_dims(frame_resized, axis=0)
        frame_resized = frame_resized / 255.0

        predictions = model.predict(frame_resized, verbose=0)

        probability_fire = predictions[0][0]
        probability_normal = predictions[0][1]

        if probability_normal > 0.5:
            classification = "normal"
        else:
            classification = "fire"

        cv2.putText(frame, f'Class: {classification}', (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

        out.write(frame)

        progress_bar.update(1)

    cap.release()
    out.release()
    progress_bar.close()

    return send_file(output_video_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
