from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Union

from bokeh.plotting import output_file, save
from bokeh.resources import ResourcesMode


def save_bokeh_figure(fig, dst: Union[str, PathLike], *, title: str = None, mode: ResourcesMode = None,
                      root_dir: Union[str, PathLike] = None) -> None:
    """Saves a Bokeh figure to an HTML document

    Args:
        fig: The Bokeh figure object to save.
        dst (str | PathLike): The destination filepath to save figure to.
        title (str, optional): Defaults to ``None``. A title for the HTML document.
        mode (str, optional) : Defaults to ``'cdn'``. how to include BokehJS.
            One of: ``'inline'``, ``'cdn'``, ``'relative(-dev)'`` or
            ``'absolute(-dev)'``. See `bokeh.resources.Resources` for more details.
        root_dir (str | PathLike, optional): Defaults to ``None``. The root directory to use for 'absolute' resources.
            This value is ignored for other resource types, e.g. ``INLINE`` or ``CDN``.
    """
    dst = Path(dst)
    title = dst.stem if title is None else title
    output_file(dst, title=title, mode=mode, root_dir=root_dir)
    save(fig)
