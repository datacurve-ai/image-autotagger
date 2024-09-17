# AutoTagger

This project is a Flask-based web application that automatically generates tags for uploaded images using a published preexisting model.

## Features

- Upload single or multiple images with a POST request
- Adjustable tag score threshold
- Response in JSON or HTML format
- REST API for integration with other services
- Asynchronous job processing with status checking

## Setup

### Installation

1. Clone the repository, create a virtual environment, activate venv, install requirements. Setup env variables if needed:

   ```
   MODEL_PATH=models/model.onnx
   PORT=8000
   DEBUG=False
   MINIMUM_WAIT_TIME_FOR_BATCHING=10
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
   curl -X POST -F "file=@/path/to/your/image.jpg" -F "threshold=0.5" -F "format=json" -F "get_id=true" http://localhost:8000/evaluate
   ```

   This command uploads an image file, sets the tag threshold to 0.5, requests JSON output, and asks for a job ID for asynchronous processing.

   Parameters:

   - `file`: The image file to upload (can be specified multiple times for batch processing)
   - `threshold`: The minimum confidence score for tags (default: 0.1)
   - `get_id`: Whether to return a job ID for asynchronous processing (default: false)

   The API will return either:
   If `get_id` is false:
   - A JSON array of objects, each containing the filename and its associated tags
   If `get_id` is true:
   - A JSON object with a `jobId` field (if `get_id` is true)

5. (Optional if done as a job) To check the status of an asynchronous job, use the `/jobstatus` endpoint:

   ```
   curl "http://localhost:8000/jobstatus?jobId=your-job-id&format=json"
   ```

   Parameters:

   - `jobId`: The job ID returned from the initial request
   - `format`: Output format, either 'json' or 'html' (default: 'json')

   This will return the job status and results if the job is complete.