from dash import callback_context


def get_trigger_id():
    return callback_context.triggered[0]["prop_id"].split(".")[0]
