"""Copy helmet.onnx and classes.json into the live component's frontend/model/ (no Streamlit here).

The browser loads them by the relative URLs model/helmet.onnx and model/classes.json, so they must sit
next to index.html. streamlit_app/model/ is the single committed copy; frontend/model/ is git-ignored and
filled by sync() (called when the component is imported, for local runs) or by
scripts/build_streamlit_repo.py (for the deployed repo).

usage: python streamlit_app/live_component/model_files.py
"""
import pathlib
import shutil

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE.parent / "model"
DST = HERE / "frontend" / "model"
FILES = ("helmet.onnx", "classes.json")


def sync(src=SRC, dst=DST):
    """Copy any missing or different-size file; returns the names copied. Never raises on a read-only
    folder (the deployed repo already contains the files)."""
    copied = []
    try:
        dst.mkdir(parents=True, exist_ok=True)
        for name in FILES:
            s, d = src / name, dst / name
            if s.is_file() and (not d.is_file() or d.stat().st_size != s.stat().st_size):
                shutil.copy2(s, d)
                copied.append(name)
    except OSError:
        pass
    return copied


if __name__ == "__main__":
    print("copied:", ", ".join(sync()) or "nothing (already up to date)")
