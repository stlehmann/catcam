from io import BytesIO
from time import sleep
from flask import Flask, make_response, url_for, render_template
import picamera
from picamera.exc import PiCameraValueError

camera = picamera.PiCamera()
camera.resolution = (640, 480)
camera.rotation=180
camera.start_preview()


buf = bytes()
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/image")
def serve_image():
    global buf
    with BytesIO() as stream:
        try:
            camera.capture(stream, "jpeg")
            buf = stream.getvalue()
        except PiCameraValueError:
            pass

    response = make_response(buf)
    response.headers.set("Content-type", "image/jpeg")
    return response


app.run("0.0.0.0", 80)
