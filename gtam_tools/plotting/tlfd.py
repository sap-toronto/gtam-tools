from __future__ import annotations

from typing import Any, Dict, Union

import pandas as pd
from balsa.routines import sort_nicely
from bokeh.core.enums import SizingModeType
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, TabPanel, Tabs
from bokeh.plotting import figure

from .common import check_df_indices, prep_figure_params, wrap_title
from .resources import SyncAxesOptions


def _core_simplify_tlfd_index(df: Union[pd.DataFrame, pd.Series], low: float = -2.0,
                         high: float = 200.0) -> Union[pd.DataFrame, pd.Series]:
    new_df = df.copy()

    new_index = [float(low)]
    for f, _ in new_df.index.values[1:-1]:
        new_index.append(float(f))
    new_index.append(float(high))
    new_df.index = new_index
    new_df.index.name = 'bin_start'

    return new_df


def _core_prep_tlfd_data(controls_df: pd.DataFrame, result_df: pd.DataFrame, data_label: str, *,
                         category_labels: Dict = None, bin_start: int = 0, bin_end: int = 200,
                         bin_step: int = 2) -> pd.DataFrame:
    # Calculate distributions
    model_tlfd_dist = result_df.div(result_df.sum(axis=0), axis=1)
    model_tlfd_dist.columns.name = data_label

    targets_tlfd_dist = controls_df.div(controls_df.sum(axis=0), axis=1)
    targets_tlfd_dist.columns.name = data_label

    df = _core_simplify_tlfd_index(model_tlfd_dist, low=bin_start - bin_step, high=bin_end).stack().to_frame(name='model')
    df['target'] = _core_simplify_tlfd_index(targets_tlfd_dist, low=bin_start - bin_step, high=bin_end).stack()
    df.reset_index(inplace=True)

    if category_labels is not None:
        df[data_label] = df[data_label].map(category_labels)

    return df


def _core_create_tlfd_plot(df: pd.DataFrame, controls_name: str, result_name: str, figure_params: Dict[str, Any], *,
                           legend_label_text_font_size: str = '11px', controls_line_colour: str = 'red',
                           controls_line_width: int = 2, **kwargs) -> figure:
    source = ColumnDataSource(df)

    p: figure = figure(**figure_params, **kwargs)
    p.line(
        source=source, x='bin_start', y='target', line_color=controls_line_colour, line_width=controls_line_width,
        legend_label=controls_name
    )
    p.vbar(
        source=source, x='bin_start', top='model', bottom=0, width=1.25, legend_label=result_name, fill_alpha=0.2
    )
    p.y_range.start = 0
    p.legend.label_text_font_size = legend_label_text_font_size

    return p


def tlfd_facet_plot(controls_df: pd.DataFrame, result_df: pd.DataFrame, data_label: str, *,
                    category_labels: Dict = None, controls_name: str = 'controls', result_name: str = 'model',
                    bin_start: int = 0, bin_end: int = 200, bin_step: int = 2, facet_col_wrap: int = 2,
                    facet_sort_order: bool = True, facet_sync_axes: SyncAxesOptions = 'both',
                    facet_max_subplot: int = 9, legend_label_text_font_size: str = '11px', figure_title: str = None,
                    plot_width: int = None, plot_height: int = None, sizing_mode: SizingModeType = None,
                    controls_line_colour: str = 'red', controls_line_width: int = 2):
    """Create an interactive Bokeh-based facet plot of TLFD diagrams using the trips table from a MicrosimData instance
    and a targets table.

    Args:
        controls_df (pd.DataFrame): A DataFrame containing TLFD targets in wide format (where rows correspond to the
            bins and columns correspond to the trips by category/group).
        result_df (pd.DataFrame): A DataFrame containing the model TLFDs in wide format (same ``controls_df``).
        data_label (str): The name to use for the data represented by the category/groups columns.
        category_labels (Dict, optional): Defaults to ``None``. Category labels used to rename the `controls_df` and
            `result_df` columns.
        controls_name (str, optional): Defaults to ``'controls'``. The name for the controls.
        result_name (str, optional): Defaults to ``'model'``. The name for the results.
        bin_start (int): Defaults is ``0``. The minimum bin value.
        bin_end (int): Defaults to ``200``. The maximum bin value.
        bin_step (int): Default is ``2``. The size of each bin.
        facet_col_wrap (int, optional): Defaults to ``2``. The number of columns to wrap subplots in the facet plot.
        facet_sort_order (bool, optional): Defaults to ``True``. A flag to render facet subplots in ascending order
            sorted by unique ``facet_col`` values.
        facet_sync_axes (SyncAxesOptions, optional): Defaults to ``'both'``. Option to sync/link facet axes. Accepts one
            of ``['both', 'x', 'y']``. Set to None to disable linked facet plot axes.
        facet_max_subplot (int, optional): Defaults to ``9``. The maximum number of facet subplots per tab. If the
            number of subplots exceed this value, a tabbed interface will be used.
        legend_label_text_font_size (str, optional): Defaults to ``'11px'``. The text size of the legend labels.
        figure_title (str, optional): Defaults to ``None``. The chart title to use.
        plot_width (int, optional): Defaults to ``None``. The desired plot width. For facet plots, this value will be
            set for each subplot.
        plot_height (int, optional): Defaults to ``None``. The desired plot height. For facet plots, this value will be
            set for each subplot.
        sizing_mode (SizingModeType, optional): Defaults to ``None``. A Bokeh SizingModeType. How will the items in the
            layout resize to fill the available space. Please refer to Bokeh documentation for acceptable values.
        controls_line_colour (str, optional): Defaults to ``'red'``. The colour to use for the control target lines.
            Accepts html colour names.
        controls_line_width (int, optional): Defaults to ``2``. The line width to use for the control target lines.

    Returns:
        A pandas DataFrame and a Bokeh figure
    """

    check_df_indices(controls_df, result_df)

    df = _core_prep_tlfd_data(
        controls_df, result_df, data_label, category_labels=category_labels, bin_start=bin_start, bin_end=bin_end,
        bin_step=bin_step
    )

    # Prepare figure formatting values
    tooltips = [('bin_start', '@bin_start'), (result_name, '@model{0.3f}'), (controls_name, '@target{0.3f}')]
    figure_params = prep_figure_params('Bin', 'Proportion', tooltips, plot_width=plot_width, plot_height=plot_height)

    # Determine number of facet plots and if groupings are needed
    n = facet_max_subplot
    facet_col_items = df[data_label].unique().tolist()
    facet_col_items = sort_nicely(facet_col_items) if facet_sort_order else facet_col_items
    facet_col_items = [facet_col_items[i * n: (i + 1) * n] for i in range((len(facet_col_items) + n - 1) // n)]

    plots = []
    for i, fc_items in enumerate(facet_col_items):
        subplots = []
        linked_axes = {}
        for j, fc in enumerate(fc_items):
            subset = df[df[data_label] == fc]  # We do it this way because CDSView doesn't support connected lines
            p = _core_create_tlfd_plot(
                subset, controls_name, result_name, figure_params,
                legend_label_text_font_size=legend_label_text_font_size, controls_line_colour=controls_line_colour,
                controls_line_width=controls_line_width, title=fc, **linked_axes
            )
            if (j == 0) and (facet_sync_axes is not None):
                if facet_sync_axes.lower() in ['x', 'both']:
                    linked_axes['x_range'] = p.x_range
                if facet_sync_axes.lower() in ['y', 'both']:
                    linked_axes['y_range'] = p.y_range
            subplots.append(p)
        plot = gridplot(
            subplots, ncols=facet_col_wrap, merge_tools=True,
            sizing_mode='stretch_both' if figure_title is None else sizing_mode
        )
        if len(facet_col_items) > 1:  # If there will be multiple tabs, convert figure into a TabPanel
            start_num = i * n + 1
            end_num = i * n + len(fc_items)
            plot = TabPanel(child=plot, title=f'Plots {start_num}-{end_num}')
        plots.append(plot)

    fig = plots[0] if len(plots) == 1 else Tabs(tabs=plots)

    if figure_title is not None:
        fig = wrap_title(fig, figure_title, sizing_mode=sizing_mode)

    return df, fig
