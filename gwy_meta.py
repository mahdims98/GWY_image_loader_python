"""
What the microscope wrote down, in a form a person can read.

A .gwy file carries a block of metadata per channel - on the AFSEM files this
program is used on, 55 entries of it: the mode, the setpoint, the scan rate,
the piezo ranges, the FPGA addresses, and, at the end, whatever the operator
typed into the comments box before starting the scan. All of it is text
written by the instrument, in the instrument's locale, so `540,000000000000
mV` is a setpoint of 540 mV and `3,333333333333 s` is a line time of 3.33 s.

Two things happen here. Reading: the entries are tidied (a decimal comma is a
decimal point, twelve zeros are not a measurement) and sorted into the handful
worth seeing at a glance - mode, setpoint, image size, scan speed, comments -
and the rest, grouped by what part of the microscope they describe. Nothing is
dropped: `sections` accounts for every key in the block, and anything this
module has not been taught about lands in "Other" in the order the file had it.

Writing: `log_block` turns a list of processing steps into the paragraph that
gets appended to the comments when a processed channel is saved, so the image
carries its own history into whatever opens it next. It is appended, never
substituted - what the operator typed at the microscope stays at the top.

No GUI toolkit and no file handling: this takes the dictionary gwy_loader
returns and gives back text.
"""

import re
from datetime import datetime


# ---------------------------------------------------------------------------
# The names the instrument uses
# ---------------------------------------------------------------------------
# One field can be called several things depending on who wrote the file, so
# each is a list of candidates and the first one present wins.

MODE_KEYS = ("Mode", "Imaging mode", "Imaging Mode", "Operating mode",
             "Scan mode")
SETPOINT_KEYS = ("Set Point", "Set point", "Setpoint", "SetPoint")
WIDTH_KEYS = ("Width", "Scan width", "X range", "Scan size X")
HEIGHT_KEYS = ("Height", "Scan height", "Y range", "Scan size Y")
XPIX_KEYS = ("Pixels/Line", "Pixels per line", "Samples/Line", "X pixels")
YPIX_KEYS = ("Lines", "Number of lines", "Y pixels")
RATE_KEYS = ("Scan Rate", "Scan rate", "Line rate")
LINETIME_KEYS = ("Time/Line", "Time per line", "Line time")
COMMENT_KEYS = ("Comments", "Comment", "User comments", "Notes")

# The advanced view in the order the instrument's parts come to mind, rather
# than the order the file happens to list them in.
GROUPS = (
    ("Session", ("User info", "Date", "Comments", "Comment", "User type",
                 "Init type")),
    ("Scan", ("Mode", "Pixels/Line", "Lines", "Width", "Height",
              "X Offset", "Y Offset", "Angle", "Scan direction",
              "Scan Rate", "Time/Line", "Scan Engine speed")),
    ("Feedback", ("P", "I", "Set Point", "Sensitivity", "Spring constant",
                  "Feedback speed", "SICM speed", "Ramp speed")),
    ("Cantilever", ("Cantilever model", "Excitation type",
                    "Tapping amplitude", "Tapping offset",
                    "Tapping frequency", "Tapping phase shift",
                    "Tapping Bandwidth", "ORT/PORT Amplitude",
                    "ORT/PORT Frequency", "ORT/PORT Offset",
                    "ZPiezo/Photothermal", "AutoSync point")),
    ("Piezo", ("Piezo name", "Piezo range X", "Piezo range Y",
               "Piezo range Z")),
    ("Electronics", ("System", "FPGA", "FPGA FB", "FPGA FB ADDR", "FPGA SE",
                     "FPGA SE ADDR", "Controller type", "Controller version",
                     "Clock", "Sampling rate", "Custom ADC/DAC", "Bits",
                     "Average band", "Average band N#")),
)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

# A value the instrument wrote as a number: an optional sign, digits, an
# optional decimal comma or point, and an optional unit stuck to the end of it.
_NUMBER = re.compile(r"^([+-]?\d+(?:[.,]\d+)?)\s*(.*)$")
# What may follow a number and still be a unit. Anything with a space, a colon
# or a comma in it is a sentence that happens to start with a digit - a date,
# a serial number, a comment - and is left exactly as it was written.
_UNIT = re.compile(r"^[A-Za-zµ°%/·^\d.\-]{0,12}$")


def tidy(value):
    """One metadata value, written the way a person would write it.

    `540,000000000000 mV` becomes `540 mV`. Only values that are a number
    followed by a unit are touched; a date, a name or a comment comes back
    exactly as the instrument left it.
    """
    text = str(value).strip()
    match = _NUMBER.match(text)
    if not match:
        return text
    number, unit = match.groups()
    if not _UNIT.match(unit):
        return text
    try:
        amount = float(number.replace(",", "."))
    except ValueError:
        return text
    # A whole number stays a whole number - a 1250000 Hz clock written as
    # 1.25e+06 Hz is tidier and less readable, which is the wrong trade.
    if amount == int(amount) and abs(amount) < 1e12:
        return f"{int(amount)} {unit}".strip()
    return f"{amount:.6g} {unit}".strip()


def _first(meta, keys):
    """The first of `keys` this block actually has, tidied."""
    for key in keys:
        if key in meta and str(meta[key]).strip():
            return tidy(meta[key])
    return None


def _split_unit(value):
    """A tidied value as (number text, unit), or (value, "") if it is not one."""
    if value is None:
        return None, ""
    parts = str(value).split(" ", 1)
    return parts[0], (parts[1] if len(parts) > 1 else "")


def image_size(meta):
    """The frame, in the units it was scanned in and in pixels."""
    width, height = _first(meta, WIDTH_KEYS), _first(meta, HEIGHT_KEYS)
    size = None
    if width and height:
        w_num, w_unit = _split_unit(width)
        h_num, h_unit = _split_unit(height)
        # Both sides are nearly always in the same unit; say it once.
        size = (f"{w_num} × {h_num} {w_unit}".strip()
                if w_unit == h_unit else f"{width} × {height}")
    elif width or height:
        size = width or height

    cols, rows = _first(meta, XPIX_KEYS), _first(meta, YPIX_KEYS)
    pixels = f"{cols} × {rows} px" if cols and rows else None
    if size and pixels:
        return f"{size}   ({pixels})"
    return size or pixels


def scan_speed(meta):
    """The line rate, with the time a line took beside it."""
    rate, per_line = _first(meta, RATE_KEYS), _first(meta, LINETIME_KEYS)
    if rate and per_line:
        return f"{rate}   ({per_line}/line)"
    return rate or (f"{per_line}/line" if per_line else None)


def comments(meta):
    """What the operator typed at the microscope, and whatever has been
    appended to it since."""
    for key in COMMENT_KEYS:
        if key in meta and str(meta[key]).strip():
            return str(meta[key]).strip()
    return ""


def compact(meta):
    """The few entries worth seeing before any of the others.

    Returns [(label, value)] with the comments last, and leaves out anything
    this file does not have rather than showing a row of blanks.
    """
    rows = [
        ("Imaging mode", _first(meta, MODE_KEYS)),
        ("Set point", _first(meta, SETPOINT_KEYS)),
        ("Image size", image_size(meta)),
        ("Scan speed", scan_speed(meta)),
    ]
    return [(label, value) for label, value in rows if value]


def sections(meta):
    """Every entry in the block, grouped.

    Returns [(group title, [(key, tidied value)])]. Keys this module has not
    been taught about are not dropped - they go into "Other" in the order the
    file listed them, which is how a metadata block from a different microscope
    still shows all of itself.
    """
    seen = set()
    out = []
    for title, keys in GROUPS:
        rows = [(key, tidy(meta[key])) for key in keys if key in meta]
        seen.update(key for key, _ in rows)
        if rows:
            out.append((title, rows))
    rest = [(key, tidy(value)) for key, value in meta.items()
            if key not in seen]
    if rest:
        out.append(("Other", rest))
    return out


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

# Gwyddion writes its strings as latin-1, so anything outside it has to go
# before the file does. Only a stray character from a pasted comment ever hits
# this; the units this program prints (µm, °) are all inside latin-1.
def latin1(text):
    """`text` with anything a .gwy file cannot hold replaced by '?'."""
    return str(text).encode("latin-1", "replace").decode("latin-1")


def log_block(steps, source=None, channel=None, when=None, tool="GWY Processor"):
    """The paragraph a save appends to the comments.

    `steps` is the list of sentences describing what was done, in order - the
    processing pipeline as it stands, which is what the saved pixels actually
    went through. A channel that carries this can be read a year later without
    the log file that was next to it.
    """
    stamp = (when or datetime.now()).strftime("%Y-%m-%d %H:%M")
    head = f"--- {tool}, {stamp}"
    if source:
        head += f", from {source}"
    if channel:
        head += f" [{channel}]"
    head += " ---"
    lines = [head]
    if steps:
        lines += [f"{i}. {step}" for i, step in enumerate(steps, 1)]
    else:
        lines.append("no processing steps applied")
    return latin1("\n".join(lines))


def with_log(meta, block, key=None):
    """A copy of `meta` with `block` appended to its comments.

    Appended, on purpose: what was typed at the microscope is the reason the
    comments field exists, and a processing history that overwrote it would be
    worse than no history at all. A file whose comments field was empty - or
    which had none - gets one.
    """
    out = dict(meta or {})
    if key is None:
        key = next((k for k in COMMENT_KEYS if k in out), COMMENT_KEYS[0])
    existing = str(out.get(key, "")).strip()
    out[key] = f"{existing}\n\n{block}" if existing else block
    return {k: latin1(v) for k, v in out.items()}
