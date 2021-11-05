import picamera
import shutil
from time import sleep

camera = picamera.PiCamera()
camera.resolution = (640, 480)
camera.rotation=180
camera.start_preview()
sleep(5)

while True:
    camera.capture("buf.jpg")
    shutil.copyfile("buf.jpg", "static/image.jpg")
    sleep(5)
