# AutoTagger

This project is a Flask-based web application that automatically generates tags for uploaded images using a published preexisting model.

## Features

- Upload single or multiple images with a POST request
- Adjustable tag score threshold
- Response in JSON or HTML format
- REST API for integration with other services

## Setup

### Installation

1. Clone the repository, create a virtual environment, activate venv, install requirements. Setup env variables if needed:

   ```
   MODEL_PATH=models/model.onnx
   PORT=8000
   DEBUG=False
   ```

2. Download the model and tags:
   ```
   mkdir -p models
   wget -O models/model.onnx https://huggingface.co/SmilingWolf/wd-v1-4-swinv2-tagger-v2/resolve/main/model.onnx
   wget -O models/selected_tags.csv https://huggingface.co/SmilingWolf/wd-v1-4-swinv2-tagger-v2/raw/main/selected_tags.csv
   ```

## Running and Usage

1. Make sure your virtual environment is active.

2. Start the Flask server:

   ```
   python app.py
   ```

3. Access the application at `http://localhost:8000` in your web browser to use the web interface for uploading and tagging images.

4. For API usage, send POST requests to `/evaluate`. Example using curl:

   ```
   curl -X POST -F "file=@/path/to/your/image.jpg" -F "threshold=0.5" -F "format=json" http://localhost:8000/evaluate
   ```

   This command uploads an image file, sets the tag threshold to 0.5, and requests JSON output.

   Parameters:

   - `file`: The image file to upload (can be specified multiple times for batch processing)
   - `threshold`: The minimum confidence score for tags (default: 0.1)
   - `format`: Output format, either 'json' or 'html' (default: 'json')

   The API will return a JSON array of objects, each containing the filename and its associated tags.
