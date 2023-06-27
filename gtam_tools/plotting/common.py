from __future__ import annotations

from bokeh.core.enums import SizingModeType
from bokeh.layouts import column
from bokeh.models import Column, Div


def wrap_title(fig, title: str, *, sizing_mode: SizingModeType = 'stretch_both') -> Column:
    return column([
        Div(text=f'<h2>{title}</h2>'),
        fig
    ], sizing_mode=sizing_mode)
