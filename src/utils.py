import json
import re
from collections import OrderedDict
from pathlib import Path

ROOT_PATH = Path(__file__).absolute().resolve().parent.parent


def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def normalize(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z ]", "", text)
    return text
