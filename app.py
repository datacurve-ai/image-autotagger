#!/usr/bin/env python
import itertools
import os
import threading
import time
import uuid
import io
from base64 import b64encode
from collections import defaultdict

import PIL.Image
from dotenv import load_dotenv
from flask import Flask, request, render_template, jsonify, abort
from werkzeug.datastructures import FileStorage

from autotagger import Autotagger

load_dotenv()
model_path = os.getenv("MODEL_PATH", "models/model.onnx")
autotagger = Autotagger(model_path)

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
app.config["JSON_PRETTYPRINT_REGULAR"] = True

MINIMUM_WAIT_TIME_FOR_BATCHING = int(os.getenv("MINIMUM_WAIT_TIME_FOR_BATCHING", 10))

thread_to_inputs = defaultdict(list)
thread_to_results = {}
thread_to_files = defaultdict(list)
batch_lock = threading.Lock()
thread_result_ready_lock = defaultdict(lambda: threading.Semaphore())
job_results = {}


def batch_manager():
    while True:
        if not thread_to_inputs:
            time.sleep(0.1)
            continue
        time.sleep(MINIMUM_WAIT_TIME_FOR_BATCHING)
        with batch_lock:
            if not thread_to_inputs:
                continue

            inputs_to_process = list(itertools.chain(*thread_to_inputs.values()))
            predictions = list(autotagger.predict(inputs_to_process))
            for thread_id, input_files in list(thread_to_inputs.items()):
                result = [pred for input_file, pred in zip(inputs_to_process, predictions) if input_file in input_files]
                job_results[thread_id] = {"status": "complete", "results": result}
                thread_to_results[thread_id] = result
                thread_to_inputs.pop(thread_id)
                thread_result_ready_lock[thread_id].release()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/evaluate", methods=["POST"])
def evaluate():
    files: list[FileStorage] = request.files.getlist("file")
    threshold = float(request.values.get("threshold", 0.1))
    general_threshold = float(request.values.get("general_threshold", threshold))
    character_threshold = float(request.values.get("character_threshold", threshold))
    output: str = request.values.get("format", "json")
    limit: int = int(request.values.get("limit", 100))
    get_id: bool = request.values.get("get_id", "false").lower() == "true"

    opened_files = []
    file_streams = []
    file_names = []

    for file in files:
        try:
            file_stream = file.stream.read()
            file_streams.append(file_stream)
            image = PIL.Image.open(io.BytesIO(file_stream))
            opened_files.append(image)
            file_names.append(file.filename)
        except PIL.UnidentifiedImageError:
            abort(400, description=f"Cannot identify image file for file {file.filename}.")
        except Exception as e:
            abort(400, description=f"Unknown exception handling opening of file image, {file.filename}: {str(e)}")

    if get_id:
        job_id: str = str(uuid.uuid4())
        thread_id: str = job_id
        thread_result_ready_lock[thread_id].acquire()
        with batch_lock:
            thread_to_inputs[thread_id].extend(opened_files)
            thread_to_files[thread_id] = list(zip(file_names, file_streams))
        job_results[job_id] = {"status": "processing"}
        if output == "html":
            return render_template("job_id_response.html", job_id=job_id)
        elif output == "json":
            return jsonify({"jobId": job_id})
        else:
            abort(400, description="Invalid output type")

    predictions = list(autotagger.predict(
        opened_files,
        general_threshold=general_threshold,
        character_threshold=character_threshold,
        limit=limit,
    ))

    if output == "html":
        files_in_base64 = [b64encode(file_stream).decode() for file_stream in file_streams]
        return render_template("evaluate.html", predictions=zip(files_in_base64, predictions))
    elif output == "json":
        predictions = [{"filename": file.filename, "tags": tags} for file, tags in zip(files, predictions)]
        return jsonify(predictions)
    else:
        abort(400, description="Invalid output type")


@app.route("/jobstatus", methods=["GET"])
def job_status():
    job_id: str = request.args.get("jobId")
    output: str = request.values.get("format", "json")
    if not job_id:
        abort(404, description="Job not found")

    job_info = job_results.get(job_id)
    if not job_info:
        abort(404, description="Job not found")

    if job_info["status"] == "complete":
        predictions = job_info["results"]
        files = thread_to_files.get(job_id, [])
        if output == "html":
            files_in_base64 = [b64encode(file_stream).decode() for _, file_stream in files]
            return render_template("evaluate.html", predictions=zip(files_in_base64, predictions))
        elif output == "json":
            predictions_json = [{"filename": filename, "tags": tags} for (filename, _), tags in zip(files, predictions)]
            return jsonify(predictions_json)
        else:
            abort(400, description="Invalid output type")
    else:
        abort(404, description="Job still processing")


if __name__ == "__main__":
    batch_processing_manager_thread = threading.Thread(target=batch_manager, daemon=True)
    batch_processing_manager_thread.start()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT") or 8000), debug=os.getenv("DEBUG") == "True")
