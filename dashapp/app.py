from dotenv import load_dotenv
load_dotenv()

import datetime
import json
import os
import pathlib
import shutil
from urllib.request import urlopen

import cv2
import dash_bootstrap_components as dbc
import numpy as np
from dash import Dash, dcc, html, callback_context
from dash.dependencies import Output, Input, State, MATCH, ALL
from dash.exceptions import PreventUpdate

from cv import process_image, frame_to_base64
import db

# query available labels from database
with db.Session() as session:
    labels = session.query(db.Label).all()

img_url = "http://192.168.1.179/image"
unlabelled_path = pathlib.Path(os.curdir) / "dashapp/assets/unlabelled"
labelled_path = pathlib.Path(os.curdir) / "dashapp/assets/labelled"

stream_img = html.Img(id="stream-img")
capture_checkbox = dbc.Switch(id="capture-checkbox", label="Capture")
update_interval = dcc.Interval(id="update-interval", interval=1000)
refresh_captured_images_button = dbc.Button("Refresh", id="refresh-captured-images-button")
capture_div = html.Div(id="capture-div", style={"margin-top": "15px", "display": "flex", "flex-wrap": "wrap"})

# dummies
label_dummy_div = html.Div(id="label-dummy-div", hidden=True)

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
    label_dummy_div,
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
            cv2.imwrite(str(unlabelled_path / fn), section)
            print(f"captured {fn}")

    return frame_to_base64(annotated_img)


@app.callback(
    Output(capture_div.id, "children"),
    Input(refresh_captured_images_button.id, "n_clicks"),
    Input(label_dummy_div.id, "children"),
)
def update_captured_images(n_clicks, children):
    children = []
    for p in unlabelled_path.iterdir():
        children.append(
            html.Div(
                [
                    html.Div(
                        html.I(className="fas fa-minus-circle"),
                        id={"type": "remove-capture-button", "index": p.name},
                        role="button",
                        className="captureRemoveDiv hide"
                    ),
                    dcc.Dropdown(
                        options=[{"label": label.name, "value": label.id} for label in labels],
                        multi=True,
                        id={"type": "select-label-dropdown", "index": p.name},
                        className="labelDropdown hide"
                    ),
                ],
                className="captureDiv",
                style={"background-image": "url('" + app.get_asset_url(f"captures/{p.name}") + "')"},
            )
        )
    return children


@app.callback(
    Output(refresh_captured_images_button.id, "n_clicks"),
    Input({"type": "remove-capture-button", "index": ALL}, "n_clicks"),
    prevent_initial_call=True
)
def remove_captured_image(n_clicks):
    triggered = callback_context.triggered[0]
    value = triggered["value"]
    id_ = ".".join(triggered["prop_id"].split(".")[:-1])
    id_ = json.loads(id_)["index"]

    if value is None:
        raise PreventUpdate

    p = unlabelled_path / id_
    p.unlink()
    return 0


@app.callback(
    Output(label_dummy_div.id, "children"),
    Input({"type": "select-label-dropdown", "index": ALL}, "value"),
    prevent_initial_call=True,
)
def label_captured_image(value):
    triggered = callback_context.triggered[0]
    value = triggered["value"]
    id_ = ".".join(triggered["prop_id"].split(".")[:-1])
    id_ = json.loads(id_)["index"]

    if value is None:
        raise PreventUpdate

    src_p = unlabelled_path / id_
    dst_p = labelled_path / value / id_
    shutil.move(src_p, dst_p)
    return 0

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=80, debug=False)