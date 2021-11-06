import base64
from urllib.request import urlopen

import cv2
import numpy as np
from dash import Dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Output, Input
from dash.exceptions import PreventUpdate

from cv import process_image, frame_to_base64

img_url = "http://192.168.1.179/image"

stream_img = html.Img(id="stream-img")
update_interval = dcc.Interval(id="update-interval", interval=1000)

app = Dash(__name__)

layout = html.Div([
    stream_img,
    update_interval,
])
app.layout = layout


@app.callback(
    Output(stream_img.id, "src"),
    Input(update_interval.id, "n_intervals")
)
def refresh_image(n_intervals):
    try:
        resp = urlopen(img_url)
        data: bytes = resp.read()
        np_arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # preprocessing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh_frame, rects = process_image(img, dilate_iterations=0)
        for (x, y, w, h) in rects[:1]:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


        # encode image to b64 for display
        # data = frame_to_base64(img)
        # data = base64.b64encode(data)
        return frame_to_base64(img)
    except:
        raise PreventUpdate

    # return f"data:image/jpg;base64,{data.decode()}"


if __name__ == "__main__":
    app.run_server(debug=True)