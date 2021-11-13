from collections import namedtuple

from dotenv import load_dotenv

load_dotenv()

import datetime
import json
import os
import pathlib
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


STREAM_URL = "http://192.168.1.179/image"
IMAGE_PATH = pathlib.Path(os.curdir) / "dashapp/assets/unlabelled"


# dash objects
stream_img = html.Img(id="stream-img")  # display actual stream
capture_checkbox = dbc.Switch(id="capture-checkbox", label="Capture")  # enable / disable capturing
update_interval = dcc.Interval(id="update-interval", interval=1000)  # interval for updating stream image
refresh_captured_images_button = dbc.Button(html.I(className="fas fa-sync"), id="refresh-captured-images-button")  # update captured images
image_list_div = html.Div(id="image-list-div")  # display captured images

image_filter_options =[
    {"label": "No Filter", "value": "no filter"},
    {"label": "Unlabeled", "value": "unlabeled"}
]
image_filter_options.extend([{"label": label.name, "value": label.id} for label in labels])
image_filter_select = dbc.Select(
    id="image-filter-select",
    options=image_filter_options,
    value=None,
)


# modals
label_modal = dbc.Modal([
    dbc.ModalHeader(dbc.ModalTitle("Add Labels")),
    dbc.ModalBody([
        dbc.Checklist(options=[{"label": label.name, "value": label.id} for label in labels], id="label-modal-checklist"),
    ]),
    dbc.ModalFooter([
        dbc.Button("OK", id="label-modal-ok-button"),
    ]),
    html.Div(id="label-modal-image-id-div", hidden=True),
], id="label-modal", is_open=False)


app = DashProxy(
    __name__,
    external_stylesheets=[dbc.themes.SKETCHY, dbc.icons.FONT_AWESOME],
    transforms=[TriggerTransform(), MultiplexerTransform()],
)

layout = html.Div([
    dbc.Navbar([
        dbc.Container([
            dbc.NavbarBrand("CatCAM"),
        ]),
    ], dark=True, color="dark"),
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
                dbc.InputGroup([
                    image_filter_select,
                    refresh_captured_images_button,
                ]),
                image_list_div,
            ]),
        ]),
    ]),
    label_modal
])
app.layout = layout


@app.callback(
    Output(stream_img.id, "src"),
    Input(update_interval.id, "n_intervals"),
    State(capture_checkbox.id, "value"),
)
def update_stream(n_intervals: int, capture: bool):
    resp = urlopen(STREAM_URL)
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
            cv2.imwrite(str(IMAGE_PATH / fn), section)

            with db.Session() as session:
                db_image = db.Image()
                db_image.name = fn
                session.add(db_image)
                session.commit()

    return frame_to_base64(annotated_img)


@app.callback(
    Output(image_list_div.id, "children"),
    Input(refresh_captured_images_button.id, "n_clicks"),
    Input(image_filter_select.id, "value"),
)
def update_images(n_btn, filter_value):
    children = []

    with db.Session() as session:
        if filter_value is None or filter_value == "no filter":
            images = session.query(db.Image).all()
        elif filter_value == "unlabeled":
            images = session.query(db.Image).filter(~db.Image.labels.any()).all()
        else:
            images = session.query(db.Image).filter(db.Image.labels.any(db.Label.id == filter_value)).all()

    for image in images:
        children.append(
            html.Div(
                [
                    html.Div(
                        html.I(className="fas fa-minus-circle"),
                        id={"type": "remove-capture-button", "index": image.name},
                        role="button",
                        className="captureRemoveDiv hide"
                    ),
                    dbc.Button(
                        "Labels",
                        className="editButton hide",
                        id={"type": "edit-capture-button", "index": image.name}
                    ),
                ],
                id={"type": "image-div", "index": image.name},
                className="captureDiv",
                style={"background-image": "url('" + app.get_asset_url(f"unlabelled/{image.name}") + "')"},
            )
        )
    return children


@app.callback(
    Output(refresh_captured_images_button.id, "n_clicks"),
    Input({"type": "remove-capture-button", "index": ALL}, "n_clicks"),
    prevent_initial_call=True
)
def remove_image(n_clicks):
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
    p = IMAGE_PATH / image_name
    p.unlink()
    return 0


@app.callback(
    Output(label_modal.id, "is_open"),
    Output("label-modal-image-id-div", "children"),
    Output("label-modal-checklist", "value"),
    Input({"type": "edit-capture-button", "index": ALL}, "n_clicks"),
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