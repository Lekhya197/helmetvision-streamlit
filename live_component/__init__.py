"""helmet_live: the Live camera component. The model runs in the visitor's browser (onnxruntime-web);
the server only serves the static files in frontend/, and no video reaches it. Only in owner mode
(report_alerts=True) does the page send alert events (a small thumbnail only with include_thumb=True).

frontend/model/ must hold helmet.onnx and classes.json; for local runs they are copied from
streamlit_app/model/ on import (model_files.sync), and the deployed repo ships them.
"""
import pathlib

import streamlit.components.v1 as components

from .model_files import sync

FRONTEND = pathlib.Path(__file__).resolve().parent / "frontend"
sync()

_component = components.declare_component("helmet_live", path=str(FRONTEND))


def helmet_live(key=None, report_alerts=False, include_thumb=False):
    """Render the live camera component.

    With report_alerts False (visitors) it returns None and nothing leaves the browser. With
    report_alerts True (owner mode) the page sends each alert to Python and this returns the latest
    one: {"id", "conf", "ts", "thumb_jpeg_base64"}; the thumbnail is null unless include_thumb is True.
    The same value comes back on every rerun until the next alert, so callers de-duplicate by id.
    """
    return _component(key=key, report_alerts=bool(report_alerts), include_thumb=bool(include_thumb),
                      default=None)
