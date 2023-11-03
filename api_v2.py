from flask import Flask, render_template, request, url_for
from pynvml import *
import os
import random
import json
import string

app = Flask(__name__)
UPLOAD_FOLDER = "static/images/uploads"
ALLOWED_EXTENSIONS = {"PNG", "png", "jpg", "JPG"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        if request.form.get("prompts"):
            prompts = request.form.get("prompts")
            name = "".join(random.choice(string.ascii_letters) for i in range(10)) + ".jpg"
            json_name = f"request_prompts/{name.split('.')[0]}.json"
            link = f"http://210.245.90.204:13000/hg_project_effect/{name}"
            data = {"name": name, "link": link, "prompts": prompts}
            warns = "Generating image. Please wait a while and view or download image with this link."
            with open(json_name, "w") as json_f:
                json.dump(data, json_f)
            return render_template("index.html", prompts=prompts, link=link, warns=warns)

    return render_template("index.html")

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        if request.files["upload_files"]:
            filename = request.files["upload_files"].filename
            style = filename.split("_")[0]
            if not os.path.exists(os.path.join(UPLOAD_FOLDER, style)):
                os.makedirs(os.path.join(UPLOAD_FOLDER, style))
            request.files["upload_files"].save(os.path.join(UPLOAD_FOLDER, style, filename))
    return render_template("upload.html")

@app.route("/information", methods=["GET", "POST"])
def infomation():
    return render_template("information.html")
if __name__ == "__main__":
    app.run(host="0.0.0.0")