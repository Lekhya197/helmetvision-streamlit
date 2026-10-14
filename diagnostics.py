"""Environment checks for the demo (pure Python, no Streamlit here): versions of the running interpreter
and differences from the pinned requirements.txt. Nothing here reads or returns a secret value.
"""
import platform
from importlib import metadata


def short_error(e, limit=300):
    """"Type: message" with the message cut to `limit` characters, for one-line logs."""
    return f"{type(e).__name__}: {str(e)[:limit]}"


def package_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def versions():
    """python, streamlit and onnxruntime versions; a missing package shows as "MISSING"."""
    return {"python": platform.python_version(),
            "streamlit": package_version("streamlit") or "MISSING",
            "onnxruntime": package_version("onnxruntime") or "MISSING"}


def pin_mismatches(path):
    """"name installed (pinned x)" for each `name==x` line of requirements.txt that differs; [] if the
    file is absent (the source tree has none, the built repo has one)."""
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        name, sep, want = line.split("#")[0].strip().partition("==")
        if sep and (package_version(name) or "not installed") != want.strip():
            out.append(f"{name} {package_version(name) or 'not installed'} (pinned {want.strip()})")
    return out


def presence(names, getter):
    """{name: True/False} for each secret name; getter(name) returns the value or None. Values are never
    kept or returned, only whether they are non-empty."""
    out = {}
    for n in names:
        try:
            v = getter(n)
        except Exception:
            v = None
        out[n] = v is not None and not isinstance(v, bool) and bool(str(v).strip())
    return out
