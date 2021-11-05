from io import BytesIO
from time import sleep
from flask import Flask, make_response, url_for, render_template
import picamera

camera = picamera.PiCamera()
camera.resolution = (640, 480)
camera.rotation=180
camera.start_preview()


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/image")
def serve_image():
    with BytesIO() as stream:
        stream = BytesIO()
        camera.capture(stream, "jpeg")
        response = make_response(stream.getvalue())

    response.headers.set("Content-type", "image/jpeg")
    return response


app.run("0.0.0.0", 80)
