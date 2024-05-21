from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Union

from bokeh.layouts import Column, GridBox
from bokeh.plotting import figure
from bokeh.models import Tabs

from bokeh.plotting import output_file, save

def save_bokeh_figure(fig: Union[Column, figure, GridBox, Tabs], dst: Union[str, PathLike], *, title: str = None):
    """Saves a bokeh figure to an HTML documnet

    Args:
        fig: The bokeh figure object to save
        dst (str | Pathlike): The destination filepath to save the figure object to.
        title (str, optional): Defaults to ''None''. A title for the HTML document
    """
    dst = Path(dst)
    title = dst.stem if title is None else title
    output_file(dst.as_posix(), title=title, mode='cdn')
    save(fig)
