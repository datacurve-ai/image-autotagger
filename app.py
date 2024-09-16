#!/usr/bin/env python
import itertools
import os
import threading
import time
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
batch_lock = threading.Lock()
thread_result_ready_lock = defaultdict(lambda: threading.Semaphore())


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
    limit = int(request.values.get("limit", 100))

    input_files = [PIL.Image.open(file.stream) for file in files]

    thread_id = threading.get_native_id()
    thread_result_ready_lock[thread_id].acquire()
    with batch_lock:
        thread_to_inputs[thread_id].extend(input_files)

    thread_result_ready_lock[thread_id].acquire()
    thread_result_ready_lock[thread_id].release()
    predictions = thread_to_results.pop(thread_id, [])

    if output == "html":
        for file in files:
            file.seek(0)

        files_in_base64 = [b64encode(file.read()).decode() for file in files]
        return render_template("evaluate.html", predictions=zip(files_in_base64, predictions))
    elif output == "json":
        predictions = [{"filename": file.filename, "tags": tags} for file, tags in zip(files, predictions)]
        return jsonify(predictions)
    else:
        abort(400, description="Invalid output type")


if __name__ == "__main__":
    batch_processing_manager_thread = threading.Thread(target=batch_manager, daemon=True)
    batch_processing_manager_thread.start()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT") or 8000), debug=os.getenv("DEBUG") == "True")
