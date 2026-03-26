from __future__ import annotations

from contextlib import contextmanager
from importlib import resources


def style_path() -> str:
    """Return the packaged Matplotlib style file path."""
    return str(resources.files("jaxqsofit").joinpath("resources/styles/jaxqsofit.mplstyle"))


@contextmanager
def use_style():
    """Apply the packaged Matplotlib style within a context manager."""
    import matplotlib.pyplot as plt

    with plt.style.context(style_path()):
        yield
