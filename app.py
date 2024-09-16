#!/usr/bin/env python

import os
from base64 import b64encode

import PIL.Image
from dotenv import load_dotenv
from flask import Flask, request, render_template, jsonify, abort
from werkzeug.datastructures import FileStorage

from autotagger import Autotagger, ProcessImageException

load_dotenv()
model_path = os.getenv("MODEL_PATH", "models/model.onnx")
autotagger = Autotagger(model_path)

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
app.config["JSON_PRETTYPRINT_REGULAR"] = True


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

    opened_files = []
    for file in files:
        try:
            opened_files.append(PIL.Image.open(file.stream))
        except PIL.UnidentifiedImageError:
            abort(400, description=f"Cannot identify image file for file {file.filename}.")
        except Exception:
            abort(400, description=f"Unknown exception handling opening of file image, {file.filename}.")

    try:
        predictions = list(autotagger.predict(
            opened_files,
            general_threshold=general_threshold,
            character_threshold=character_threshold,
            limit=limit,
        ))
    except ProcessImageException as process_image_exception:
        abort(400, description=f"Error processing image file, {process_image_exception.message}.")

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
    app.run(host="0.0.0.0", port=int(os.getenv("PORT") or 8000), debug=os.getenv("DEBUG") == "True")
