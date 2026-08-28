"""Optional visualization imports (``pymcfs[viz]``)."""

from __future__ import annotations


def require_plotly():
    """Import ``plotly.graph_objects`` or raise a clear install hint.

    Returns
    -------
    module
        ``plotly.graph_objects``.

    Raises
    ------
    ImportError
        If Plotly is not installed (``pymcfs[viz]``).
    """
    try:
        import plotly.graph_objects as go
    except ImportError as e:
        raise ImportError(
            "plotly is required for this plotting helper. "
            "Install with: pip install 'pymcfs[viz]'  (or: uv sync --extra viz)"
        ) from e
    return go


def require_matplotlib_pyplot():
    """Import ``matplotlib.pyplot`` or raise a clear install hint.

    Returns
    -------
    module
        ``matplotlib.pyplot``.

    Raises
    ------
    ImportError
        If Matplotlib is not installed (``pymcfs[viz]``).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for this plotting helper. "
            "Install with: pip install 'pymcfs[viz]'  (or: uv sync --extra viz)"
        ) from e
    return plt
