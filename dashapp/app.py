from collections import namedtuple

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
from dash import Dash, dcc, html, callback_context, no_update
from dash.dependencies import Output, Input, State, MATCH, ALL
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import DashProxy, TriggerTransform, MultiplexerTransform


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

# modals
label_modal = dbc.Modal([
    dbc.ModalHeader(dbc.ModalTitle("Add Labels")),
    dbc.ModalBody([
        dbc.Checklist(options=[{"label": label.name, "value": label.id} for label in labels], id="label-modal-checklist"),
    ]),
    dbc.ModalFooter([
        dbc.Button("OK", id="label-modal-ok-button"),
    ]),
    html.Div(id="label-modal-image-id-div", hidden=False),
], id="label-modal", is_open=False)

# dummies
label_dummy_div = html.Div(id="label-dummy-div", hidden=True)

app = DashProxy(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    transforms=[TriggerTransform(), MultiplexerTransform()],
)

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
    label_modal
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

            with db.Session() as session:
                db_image = db.Image()
                db_image.name = fn
                session.add(db_image)
                session.commit()

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
                ],
                id={"type": "image-div", "index": p.name},
                className="captureDiv",
                style={"background-image": "url('" + app.get_asset_url(f"unlabelled/{p.name}") + "')"},
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
    image_name = id_ = json.loads(id_)["index"]

    if value is None:
        raise PreventUpdate

    # remove image from database
    with db.Session() as session:
        db_image = session.query(db.Image).filter_by(name=image_name).one()
        session.delete(db_image)
        session.commit()

    # remove image file
    p = unlabelled_path / image_name
    p.unlink()
    return 0


@app.callback(
    Output(label_modal.id, "is_open"),
    Output("label-modal-image-id-div", "children"),
    Output("label-modal-checklist", "value"),
    Input({"type": "image-div", "index": ALL}, "n_clicks"),
    Input("label-modal-ok-button", "n_clicks"),
    State("label-modal-checklist", "value"),
    State("label-modal-image-id-div", "children"),
    prevent_initial_call=True,
)
def open_label_modal(n_images, n_ok_button, checked_label_ids, image_name):

    ReturnValues = namedtuple("ReturnValues", ["modal_is_open", "image_name", "checked_label_ids"])

    trigger = callback_context.triggered[0]
    trigger_id = trigger["prop_id"]
    trigger_value = trigger["value"]

    # OK-button pressed: Add labels to image
    if trigger_id.split(".")[0] == "label-modal-ok-button":
        with db.Session() as session:
            db_img = session.query(db.Image).filter_by(name=image_name).one()
            db_labels = session.query(db.Label).filter(db.Label.id.in_(checked_label_ids)).all()
            for label in db_labels:
                db_img.labels.append(label)
            session.add(db_img)
            session.commit()
        return ReturnValues(modal_is_open=False, image_name=no_update, checked_label_ids=no_update)

    # Image clicked: Open modal
    trigger_id = ".".join(trigger["prop_id"].split(".")[:-1])
    image_name = trigger_id = json.loads(trigger_id)["index"]  # the trigger id is the filename of the image

    with db.Session() as session:
        db_image = session.query(db.Image).filter_by(name=image_name).one()
        label_ids = [label.id for label in db_image.labels]

    if trigger_value is None:
        raise PreventUpdate

    return ReturnValues(modal_is_open=True, image_name=image_name, checked_label_ids=label_ids)


if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=80, debug=False)