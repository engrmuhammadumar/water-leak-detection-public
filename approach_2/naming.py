# naming.py
import re
_non_alnum = re.compile(r"[^0-9a-z]+")
_spaces    = re.compile(r"\s+")
def normalize_stem(stem: str) -> str:
    s = stem.lower()
    s = _non_alnum.sub(" ", s)
    s = _spaces.sub(" ", s).strip()
    return s
