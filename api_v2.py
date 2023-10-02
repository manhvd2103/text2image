from flask import Flask, render_template, request, url_for
from pynvml import *
import os
import random
import json
import string
from txt2img import txt2img

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
            json_name = f"json_files/{name.split('.')[0]}.json"
            link = f"http://210.245.90.204:13000/hg_project_effect/{name}"
            data = {"name": name, "link": link, "prompts": prompts}
            warns = "GPUs is running please wait a while. You should save this link and download images in some minute ago."
            nvmlInit()
            h = nvmlDeviceGetHandleByIndex(0)
            info = nvmlDeviceGetMemoryInfo(h)
            if info.used / info.total > 0.4:
                with open(json_name, "w") as json_f:
                    json.dump(data, json_f)
                return render_template("index.html", prompts=prompts, link=link, warns=warns)
            else:
                path = txt2img(prompt=prompts, name=name)
                return render_template("index.html", img_path=path, prompts=prompts)
    return render_template("index.html")

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        if request.files["upload_files"]:
            filename = request.files["upload_files"].filename
            request.files["upload_files"].save(os.path.join(UPLOAD_FOLDER, filename))
    return render_template("upload.html")

@app.route("/information", methods=["GET", "POST"])
def infomation():
    return render_template("information.html")
if __name__ == "__main__":
    app.run(host="0.0.0.0")