"""Optional integrations — every one of these is a nicety, never a requirement.

Each import is attempted once, at package import time, and its absence is recorded
rather than raised. Nothing in the client may hard-fail because Pillow or Pygments
is missing.
"""

import shutil

try:  # pragma: no cover - depends on environment
    from PIL import Image as _PILImage  # type: ignore
except Exception:  # pragma: no cover
    _PILImage = None

try:  # pragma: no cover
    import pygments  # type: ignore
    from pygments import highlight as _pygments_highlight  # type: ignore
    from pygments.formatters import Terminal256Formatter as _PygTerm  # type: ignore
    from pygments.lexers import get_lexer_by_name as _pyg_lexer  # type: ignore
except Exception:  # pragma: no cover
    pygments = None
    _pygments_highlight = None
    _PygTerm = None
    _pyg_lexer = None

try:  # pragma: no cover
    import pytesseract  # type: ignore
except Exception:  # pragma: no cover
    pytesseract = None


def optional_features() -> dict:
    """What the environment offers on top of the standard library."""
    return {'pillow': _PILImage is not None, 'pygments': pygments is not None,
            'pytesseract': pytesseract is not None,
            'tesseract': shutil.which('tesseract') is not None}
