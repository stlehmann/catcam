import datetime
import os
import pathlib
from urllib.request import urlopen

import cv2
import dash_bootstrap_components as dbc
import numpy as np
from dash import Dash, dcc, html
from dash.dependencies import Output, Input, State
from dash.exceptions import PreventUpdate

from cv import process_image, frame_to_base64

labels = ["unknown", "cat", "hooman"]
img_url = "http://192.168.1.179/image"
capture_path = pathlib.Path(os.curdir) / "dashapp" / "assets" / "captures"

stream_img = html.Img(id="stream-img")
capture_checkbox = dbc.Switch(id="capture-checkbox", label="Capture")
update_interval = dcc.Interval(id="update-interval", interval=1000)
refresh_captured_images_button = dbc.Button("Refresh", id="refresh-captured-images-button")
capture_div = html.Div(id="capture-div", style={"margin-top": "15px", "display": "flex", "flex-wrap": "wrap"})

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME])

layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col([
                stream_img,
            ]),
        ]),
        dbc.Row([
            dbc.Col([
                capture_checkbox,
            ]),
        ]),
        dbc.Row([
            dbc.Col([
                update_interval,
            ]),
        ]) ,
        dbc.Row([
           dbc.Col([
               refresh_captured_images_button,
               capture_div,
           ]),
        ]),
    ]),
])
app.layout = layout


@app.callback(
    Output(stream_img.id, "src"),
    Input(update_interval.id, "n_intervals"),
    State(capture_checkbox.id, "value"),
)
def refresh_image(n_intervals: int, capture: bool):
    resp = urlopen(img_url)
    img = cv2.imdecode(np.frombuffer(resp.read(), np.uint8), cv2.IMREAD_COLOR)

    # preprocessing
    annotated_img = img.copy()
    thresh_frame, rects = process_image(img, dilate_iterations=1)
    if len(rects) > 2:
        rects = rects[:2]
    for (x, y, w, h) in rects:
        cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if capture:
        for i, (x, y, w, h) in enumerate(rects, start=1):
            section = img[y:y+h, x:x+w]
            dt = datetime.datetime.now()
            fn = f"{dt:%Y-%m-%d_%H-%M-%S}_{i:02}.jpg"
            cv2.imwrite(str(capture_path / fn), section)
            print(f"captured {fn}")

    return frame_to_base64(annotated_img)


@app.callback(
    Output(capture_div.id, "children"),
    Input(refresh_captured_images_button.id, "n_clicks"),
)
def update_captured_images(n_clicks):
    children = []
    for p in capture_path.iterdir():
        children.append(
            html.Div(
                html.Div(html.A(html.I(className="fas fa-minus-circle"), href="#"), className="captureRemoveDiv hide"),
                className="captureDiv",
                style={"background-image": "url('" + app.get_asset_url(f"captures/{p.name}") + "')"},
            )
        )
    return children


if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=80, debug=True)