from __future__ import annotations

from typing import Any, Dict, Hashable, List, Tuple, Union

import numpy as np
import pandas as pd
from balsa.routines import sort_nicely
from bokeh.core.enums import SizingModeType
from bokeh.layouts import gridplot
from bokeh.models import (CDSView, ColumnDataSource, GroupFilter, Slope,
                          TabPanel, Tabs)
from bokeh.palettes import Category20
from bokeh.plotting import figure

from .common import (check_df_indices, check_ref_label, prep_figure_params,
                     wrap_title)
from .resources import SyncAxesOptions


def _core_get_scatterplot_data(controls_df: pd.DataFrame, result_df: pd.DataFrame, data_label: str,
                               ref_label: List[str], *, category_labels: Dict = None, controls_name: str = 'controls',
                               result_name: str = 'model', totals_in_titles: bool = True,
                               filter_zero_rows: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = controls_df.stack()
    df.index.names = [*ref_label, data_label]
    df = df.to_frame(name=controls_name)

    df[result_name] = result_df.stack()
    df[result_name].fillna(0, inplace=True)

    if filter_zero_rows:
        df = df[df.sum(axis=1) > 0].copy()

    df.reset_index(inplace=True)

    if category_labels is not None:
        df[data_label] = df[data_label].map(category_labels)

    fig_df = df.copy()
    if totals_in_titles:
        label_totals = fig_df.groupby(data_label)[[controls_name, result_name]].sum()
        label_totals['label'] = label_totals.index + f' ({controls_name}=' + label_totals[controls_name].map(
            '{:,.0f}'.format) + f', {result_name}=' + label_totals[result_name].map('{:,.0f}'.format) + ')'
        fig_df[data_label] = fig_df[data_label].map(label_totals['label'])

    return df, fig_df


def _core_create_scatterplot(fig_df: pd.DataFrame, figure_params: Dict[str, Any], glyph_params: Dict[str, Any], *,
                             glyph_col: str = None, glyph_color_palette: List[str] = None, glyph_legend: bool = True,
                             glyph_legend_location: str = 'bottom_right', glyph_legend_label_text_font_size: str = '11px',
                             identity_line: bool = True, identity_color: str = 'red', identity_width: int = 2,
                             **kwargs) -> figure:
    p: figure = figure(**figure_params, **kwargs)
    if glyph_col is None:
        p.circle(**glyph_params)
    else:  # Iterate through unique `glyph_col` values to use interactive legend feature
        for i, gc in enumerate(sort_nicely(fig_df[glyph_col].unique().tolist())):
            glyph_group_filter = GroupFilter(column_name=glyph_col, group=gc)
            view = CDSView(filter=glyph_group_filter)
            p.circle(view=view, legend_label=gc, color=glyph_color_palette[i], **glyph_params)

        # Apply legend settings
        p.legend.visible = glyph_legend
        p.legend.title = glyph_col
        p.legend.location = glyph_legend_location
        p.legend.label_text_font_size = glyph_legend_label_text_font_size
        p.legend.click_policy = 'hide'

    if identity_line:
        slope = Slope(
            gradient=1, y_intercept=0, line_color=identity_color, line_dash='dashed', line_width=identity_width
        )
        p.add_layout(slope)

    return p


def scatterplot_comparison(controls_df: pd.DataFrame, result_df: pd.DataFrame, data_label: str, *,
                           ref_label: Union[str, List[str]] = None, category_labels: Dict = None,
                           controls_name: str = 'controls', result_name: str = 'model', size: float = 7.5,
                           fill_alpha: float = 0.2, facet_col: str = None, facet_col_wrap: int = 2,
                           facet_sort_order: bool = True, facet_sync_axes: SyncAxesOptions = 'both',
                           facet_max_subplot: int = 9, hover_col: Union[str, List[str]] = None, glyph_col: str = None,
                           glyph_legend: bool = True, glyph_legend_location: str = 'bottom_right',
                           glyph_legend_label_text_font_size: str = '11px', figure_title: str = None,
                           plot_width: int = None, plot_height: int = None, sizing_mode: SizingModeType = None,
                           identity_line: bool = True, identity_color: str = 'red', identity_width: int = 2,
                           color_palette: Dict[int, Any] = Category20, calc_pct_diff: bool = True,
                           totals_in_titles: bool = True, filter_zero_rows: bool = True):
    """Creates an interactive Bokeh-based scatter plot to compare data.

    Args:
        controls_df (pd.DataFrame): A DataFrame containing control values. Must be in wide-format where rows represent
            a reference (e.g. count station, TAZ, geography, etc.) and columns represent the data categories.
        result_df (pd.DataFrame): A DataFrame containing modelled values. Uses the same format as `controls_df`.
        data_label (str): The name to use for the data represented by the `controls_df` and `result_df` columns.
        ref_label (str | List[str], optional): Defaults to ``None``. The name(s) corresponding to the ``controls_df``
            and ``result_df`` indices. The function will try to infer the name(s) from indices of the source DataFrames.
            If the indices of the DataFrames are not set, then values must be set for this parameter, otherwise an error
            will be raised. If providing a value to this parameter and the indices of the source DataFrames are
            MultiIndex objects, then the provided value must be a list of strings.
        category_labels (Dict, optional): Defaults to ``None``. Category labels used to rename the `controls_df` and
            `result_df` columns.
        controls_name (str, optional): Defaults to ``'controls'``. The name for the controls.
        result_name (str, optional): Defaults to ``'model'``. The name for the results.
        size (float, optional): Defaults to ``7.5``. The size of the scatter plot points.
        fill_alpha (float, optional): Defaults to ``0.2``. The opacity of the point fill.
        facet_col (str, optional): Defaults to ``None``. The name of the column to use for creating a facet plot.
        facet_col_wrap (int, optional): Defaults to ``2``. The number of columns to wrap subplots in the facet plot.
        facet_sort_order (bool, optional): Defaults to ``True``. A flag to render facet subplots in ascending order
            sorted by unique ``facet_col`` values.
        facet_sync_axes (SyncAxesOptions, optional): Defaults to ``'both'``. Option to sync/link facet axes. Accepts one
            of ``['both', 'x', 'y']``. Set to None to disable linked facet plot axes.
        facet_max_subplot (int, optional): Defaults to ``9``. The maximum number of facet subplots per tab. If the
            number of subplots exceed this value, a tabbed interface will be used.
        hover_col (str | List[str], optional): Defaults to ``None``. The column names to display in the plot tooltips.
        glyph_col (str, optional): Defaults to ``None``. The name of the column to use for glyph coloring. A standard
            color palette will be mapped to unique ``glyph_col`` values.
        glyph_legend (bool, optional): Defaults to ``True``. A flag to enable/disable the legend if ``glyph_col`` is
            set. The legend will be included in each plot/facet subplot.
        glyph_legend_location (str, optional): Defaults to ``'bottom_right'``. The location of the glyph legend in each
            plot/facet subplot. Please refer to the Bokeh ``Legend`` documentation for acceptable values.
        glyph_legend_label_text_font_size (str, optional): Defaults to ``'11px'``. The text size of the legend labels.
        figure_title (str, optional): Defaults to ``None``. The chart title to use.
        plot_width (int, optional): Defaults to ``None``. The desired plot width. For facet plots, this value will be
            set for each subplot.
        plot_height (int, optional): Defaults to ``None``. The desired plot height. For facet plots, this value will be
            set for each subplot.
        sizing_mode (SizingModeType, optional): Defaults to ``None``. A Bokeh SizingModeType. How will the items in the
            layout resize to fill the available space. Please refer to Bokeh documentation for acceptable values.
        identity_line (bool, optional): Defaults to ``True``. A flag to include an identity (1:1) line in the
            scatter plot.
        identity_color (str, optional): Defaults to ``'red'``. The color to use for the identity line. Accepts html
            color names.
        identity_width (int, optional): Defaults to ``2``. The line width to use for the identity line.
        color_palette (Dict[str, Any], optional): Defaults to ``Category20``. The Bokeh color palette to use.
        calc_pct_diff (bool, optional): Defaults to ``True``. Include percent difference calculation in DataFrame output
        totals_in_titles (bool, optional): Defaults to ``True``. Include the control and result totals in plot title.
        filter_zero_rows (bool, optional): Defaults to ``True``. Filter out comparisons where controls and results are
            both zeros.
    """
    check_df_indices(controls_df, result_df)
    ref_label = check_ref_label(controls_df, result_df, ref_label)

    if hover_col is None:
        hover_col = []
    if isinstance(hover_col, Hashable):
        hover_col = [hover_col]
    elif isinstance(hover_col, List):
        pass
    else:
        raise RuntimeError('Invalid data type provided for `hover_col`')

    df, fig_df = _core_get_scatterplot_data(
        controls_df, result_df, data_label, ref_label, category_labels=category_labels, controls_name=controls_name,
        result_name=result_name, totals_in_titles=totals_in_titles, filter_zero_rows=filter_zero_rows
    )

    selected_color_palette = None
    if glyph_col is not None:
        n_colors = max(len(fig_df[glyph_col].unique()), 3)
        selected_color_palette = color_palette[n_colors]

    # Prepare figure formatting values
    tooltips = [(c, '@{%s}' % c) for c in hover_col]
    tooltips += [(controls_name, '@{%s}{0,0.0}' % controls_name), (result_name, '@{%s}{0,0.0}' % result_name)]
    figure_params = prep_figure_params(
        controls_name, result_name, tooltips, plot_width=plot_width, plot_height=plot_height
    )

    # Plot figure
    if facet_col is None:
        source = ColumnDataSource(fig_df)
        glyph_params = {
            'source': source, 'x': controls_name, 'y': result_name, 'size': size, 'fill_alpha': fill_alpha,
            'hover_color': 'red'
        }
        fig = _core_create_scatterplot(
            fig_df, figure_params, glyph_params, glyph_col=glyph_col, glyph_color_palette=selected_color_palette,
            glyph_legend=glyph_legend, glyph_legend_location=glyph_legend_location,
            glyph_legend_label_text_font_size=glyph_legend_label_text_font_size, identity_line=identity_line,
            identity_color=identity_color, identity_width=identity_width,
            sizing_mode='stretch_both' if figure_title is None else sizing_mode
        )
    else:
        # Determine number of facet plots and if groupings are needed
        n = facet_max_subplot
        facet_col_items = fig_df[facet_col].unique().tolist()
        facet_col_items = sort_nicely(facet_col_items) if facet_sort_order else facet_col_items
        facet_col_items = [facet_col_items[i * n: (i + 1) * n] for i in range((len(facet_col_items) + n - 1) // n)]

        plots = []
        for i, fc_items in enumerate(facet_col_items):
            subplots = []
            linked_axes = {}
            for j, fc in enumerate(fc_items):
                fig_sub_df = fig_df[fig_df[facet_col] == fc].copy()
                source = ColumnDataSource(fig_sub_df)
                glyph_params = {
                    'source': source, 'x': controls_name, 'y': result_name, 'size': size, 'fill_alpha': fill_alpha,
                    'hover_color': 'red'
                }
                p = _core_create_scatterplot(
                    fig_sub_df, figure_params, glyph_params, glyph_col=glyph_col, glyph_color_palette=color_palette,
                    glyph_legend=glyph_legend, glyph_legend_location=glyph_legend_location,
                    glyph_legend_label_text_font_size=glyph_legend_label_text_font_size, identity_line=identity_line,
                    identity_color=identity_color, identity_width=identity_width, title=fc, **linked_axes
                )
                if (j == 0) and (facet_sync_axes is not None):
                    if facet_sync_axes.lower() in {'x', 'both'}:
                        linked_axes['x_range'] = p.x_range
                    if facet_sync_axes.lower() in {'y', 'both'}:
                        linked_axes['y_range'] = p.y_range
                subplots.append(p)

            plot = gridplot(
                subplots, ncols=facet_col_wrap, merge_tools=True,
                sizing_mode='stretch_both' if figure_title is None else sizing_mode
            )
            if len(facet_col_items) > 1:  # If there will be multiple tabs, convert figure into a TabPanel
                start_num = i * n + 1
                end_num = i * n + len(fc_items)
                plot = TabPanel(child=fig, title=f'Plots {start_num}-{end_num}')
            plots.append(plot)

        fig = plots[0] if len(plots) == 1 else Tabs(tabs=plots)

    if figure_title is not None:
        fig = wrap_title(fig, figure_title, sizing_mode=sizing_mode)

    if calc_pct_diff:
        df['pct_diff'] = (df[result_name] - df[controls_name]) / df[controls_name] * 100
        df['pct_diff'] = df['pct_diff'].replace([np.inf, -np.inf], np.nan)

    return df, fig
