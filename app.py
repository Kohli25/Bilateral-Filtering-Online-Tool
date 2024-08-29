from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
import os
from werkzeug.utils import secure_filename
from datetime import datetime
from algorithm import bilateral_cross_filtering  # Your image processing algorithm

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['OUTPUT_FOLDER'] = 'static/output/'
socketio = SocketIO(app)  # Initialize SocketIO

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_images():
    """Handle image uploads and processing."""
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': 'Both images are required'}), 400

    image1 = request.files['image1']
    image2 = request.files['image2']

    if image1.filename == '' or image2.filename == '':
        return jsonify({'error': 'Both images must have a filename'}), 400

    if image1 and allowed_file(image1.filename) and image2 and allowed_file(image2.filename):
        filename1 = secure_filename(image1.filename)
        filename2 = secure_filename(image2.filename)
        image1_path = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        image2_path = os.path.join(app.config['UPLOAD_FOLDER'], filename2)

        image1.save(image1_path)
        image2.save(image2_path)

        # Run the algorithm and process images
        output_path, info = bilateral_cross_filtering(image1_path, image2_path, socketio)

        return jsonify({'processedImageUrl': output_path, 'info': info}),200

    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/static/output/<filename>')
def serve_output_image(filename):
    """Serve the processed output image."""
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection."""
    print("Client connected")

@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection."""
    print("Client disconnected")

if __name__ == '__main__':
    socketio.run(app, debug=True)
