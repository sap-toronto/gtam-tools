from __future__ import annotations

from typing import Any, Dict, Hashable, List, Tuple, Union

import numpy as np
import pandas as pd

from balsa.routines import sort_nicely
from bokeh.layouts import Column, GridBox, gridplot
from bokeh.models import ColumnDataSource, Panel, Slope, Tabs
from bokeh.palettes import Category20, Category10, Category20b, Category20c, Set1, Set2, Set3
from bokeh.plotting import figure

from common import (check_df_indices, check_ref_label, prep_figure_params, wrap_figure_title)

def _core_get_scatterplot_data(controls_df: pd.DataFrame, result_df: pd.DataFrame, data_label: str, ref_label: Union[str, List[str]] = None, category_labels: Dict = None,
                               controls_name: str = 'controls', result_name: str = 'model', totals_in_titles: bool = True, filter_zero_rows: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepares the scatter plot data for plotting

    Args:
        controls_df (pd.DataFrame): A DataFrame containing control values. Must be in the form where rows represent a reference and columns represent the data categories.
        result_df (pd.DataFrame): A DataFrame containing modelled values. Must be the same format as 'controls_df'.
        data_label (str): The name for the data represented by the 'controls_df' and 'result_df' columns.
        ref_label (Union[str, List[str]], optional): Defaults to ''None''. The name(s) corresponding to the 'controls_df' and 'result_df' indices.
        category_labels (Dict, optional): Defaults to ''None''. Category labels used to rename the 'controls__df' and 'results_df' columns. 
        controls_name (str, optional): Defaults to ''controls''. The name for the controls.
        result_name (str, optional): Defaults to ''model''. The name for the results
        totals_in_titles (bool, optional): Defaults to ''True''. Include the control and result totals in the plot title. 
        filter_zero_rows (bool, optional): Defaults to ''True''. Filter out comparisons where both controls and results are zero.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]
    """
    df = controls_df.stack()
    df.index.names = [*ref_label, data_label]
    df = df.to_frame(name=controls_name)

    df[result_name] = result_df.stack()
    df.fillna({result_name : 0}, inplace = True)

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


def _core_create_scatterplot(fig_df: pd.DataFrame, p: figure, glyph_params: Dict[str, Any], *, glyph_col: str = None, color_palette: List[str] = Category20,
                             glyph_legend: bool = True, glyph_legend_location: str = 'bottom_right', glyph_legend_label_text_font_size: str = '11px') -> figure:
    """Creates and plots scatterplot

    Args:
        fig_df (pd.DataFrame): The DataFrame containing data to be plotted 
        source (ColumnDataSource): The Column Data Source to use when setting the points for the scatterplot
        p (figure): The figure where these points are placed on a scatterplot
        glyph_params (Dict[str, Any]): Parameters of each point on the plot
        glyph_col (str, optional): Default to ''None''. The name of the column to use for glyph coloring.
        color_palette (List[str], optional): Defaults to ''Category20''. The Bokeh color palette to use. 
        glyph_legend (bool, optional): Defaults to ''True''. Enables or disables a legend if ''glyph_col'' is set. 
        glyph_legend_location (str, optional): Defaults to ''bottom_right''. The location of the glyph legend in each plot/facet subplot. 
        glyn_legend_label_text_font_size (str, optional): Defaults to ''11px''. The size of the text of the legend labels

    Returns:
        A Bokeh figure
    """
    def apply_legend_settings(p_: figure):
        p_.legend.visible = glyph_legend
        p_.legend.title = glyph_col
        p_.legend.location = glyph_legend_location
        p_.legend.label_text_font_size = glyph_legend_label_text_font_size
        p_.legend.click_policy = 'hide'

    if glyph_col is None:  # Single glyphs
        source = ColumnDataSource(fig_df)
        p.scatter(**glyph_params, source = source)
    else:  # Iterate through unique `glyph_col` values to use interactive legend feature
        for j, gc in enumerate(sorted(fig_df[glyph_col].unique())):
            subset_df = fig_df[fig_df[glyph_col] == gc]
            subset_source = ColumnDataSource(subset_df)
            p.scatter(source = subset_source, legend_label = gc, color = color_palette[j], **glyph_params)
        apply_legend_settings(p)
    
    return p


def scatterplot_comparison(controls_df: pd.DataFrame, result_df: pd.DataFrame, data_label: str, *,
                           ref_label: Union[str, List[str]] = None, category_labels: Dict = None,
                           controls_name: str = 'controls', result_name: str = 'model', size: float = 7.5,
                           fill_alpha: float = 0.2, facet_col: str = None, facet_col_wrap: int = 2,
                           facet_sort_order: bool = True, facet_sync_axes: str = 'both', facet_max_subplot: int = 9,
                           hover_col: Union[str, List[str]] = None, glyph_col: str = None, glyph_legend: bool = True,
                           glyph_legend_location: str = 'bottom_right', glyph_legend_label_text_font_size: str = '11px',
                           figure_title: str = None, height: int = None, identity_line: bool = True,
                           identity_colour: str = 'red', identity_width: int = 2,
                           color_palette: Dict[int, Any] = Category20, calc_pct_diff: bool = True,
                           totals_in_titles: bool = True, filter_zero_rows: bool = True
                           ) -> Tuple[pd.DataFrame, Union[Column, figure, GridBox, Tabs]]:
    """Creates an interactive Bokeh-based scatter plot to compare data.

    Args:
        controls_df (pd.DataFrame): A DataFrame containing control values. Must be in wide-format where rows represent
            a reference (e.g. count station, TAZ, geography, etc.) and columns represent the data categories.
        result_df (pd.DataFrame): A DataFrame containing modelled values. Uses the same format as `controls_df`.
        data_label (str): The name to use for the data represented by the `controls_df` and `result_df` columns.
        ref_label (Union[str, List[str]], optional): Defaults to ``None``. The name(s) corresponding to the
            ``controls_df`` and ``result_df`` indices. The function will try to infer the name(s) from indices of the
            source DataFrames. If the indicies of the DataFrames are not set, then values must be set for this
            parameter, otherwise an error will be raised. If providing a value to this parameter and the indices of the
            source DataFrames are MultiIndex objects, then the provided value must be a list of strings.
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
        facet_sync_axes (str, optional): Defaults to ``'both'``. Option to sync/link facet axes. Accepts one of
            ``['both', 'x', 'y']``. Set to None to disable linked facet plot axes.
        facet_max_subplot (int, optional): Defaults to ``9``. The maximum number of facet subplots per tab. If the
            number of subplots exceed this value, a tabbed interface will be used.
        hover_col (Union[str, List[str]], optional): Defaults to ``None``. The column names to display in the plot
            tooltips.
        glyph_col (str, optional): Defaults to ``None``. The name of the column to use for glyph coloring. A standard
            color palette will be mapped to unique ``glyph_col`` values.
        glyph_legend (bool, optional): Defaults to ``True``. A flag to enable/disable the legend if ``glyph_col`` is
            set. The legend will be included in each plot/facet subplot.
        glyph_legend_location (str, optional): Defaults to ``'bottom_right'``. The location of the glyph legend in each
            plot/facet subplot. Please refer to the Bokeh ``Legend`` documentation for acceptable values.
        glyph_legend_label_text_font_size (str, optional): Defaults to ``'11px'``. The text size of the legend labels.
        figure_title (str, optional): Defaults to ``None``. The chart title to use.
        height (int, optional): Defaults to ``None``. The desired plot height. For facet plots, this value will be
            set for each subplot.
        identity_line (bool, optional): Defaults to ``True``. A flag to include an identity (1:1) line in the
            scatter plot.
        identity_colour (str, optional): Defaults to ``'red'``. The colour to use for the identity line. Accepts html
            colour names.
        identity_width (int, optional): Defaults to ``2``. The line width to use for the identity line.
        color_palette (Dict[str, Any], optional): Defaults to ``Category20``. The Bokeh color palette to use.
        calc_pct_diff (bool, optional): Defaults to ``True``. Include percent difference calculation in DataFrame output
        totals_in_titles (bool, optional): Defaults to ``True``. Include the control and result totals in plot title.
        filter_zero_rows (bool, optional): Defaults to ``True``. Filter out comparisons where controls and results are
            both zeros.

    Returns:
        Tuple[pd.DataFrame, Union[Column, figure, GridBox, Tabs]]
    """
    check_df_indices(controls_df, result_df)

    ref_label = check_ref_label(ref_label, controls_df, result_df)

    if hover_col is None:
        hover_col = []
    if isinstance(hover_col, Hashable):
        hover_col = [hover_col]
    elif isinstance(hover_col, List):
        pass
    else:
        raise RuntimeError('Invalid data type provided for `ref_label`')

    df, fig_df = _core_get_scatterplot_data(controls_df, result_df, data_label, ref_label, category_labels = category_labels, controls_name = controls_name,
                                            result_name = result_name, totals_in_titles= totals_in_titles, filter_zero_rows= filter_zero_rows)

    if glyph_col is not None:
        n_colors = max(len(fig_df[glyph_col].unique()), 3)
        color_palette = color_palette[n_colors]

    # Prepare figure formatting values
    source = ColumnDataSource(fig_df)
    tooltips = [(c, '@{%s}' % c) for c in hover_col]
    tooltips += [(controls_name, '@{%s}{0,0.0}' % controls_name), (result_name, '@{%s}{0,0.0}' % result_name)]
    figure_params = prep_figure_params(controls_name, result_name, tooltips, height)

    glyph_params = {
        'x': controls_name, 'y': result_name, 'size': size, 'fill_alpha': fill_alpha,
        'hover_color': 'red'
    }

    slope = Slope(gradient=1, y_intercept=0, line_color=identity_colour, line_dash='dashed', line_width=identity_width)

    # Plot figure
    if facet_col is None:  # Basic plot
        if height is None:
            p = figure(sizing_mode='stretch_both', **figure_params)
        else:
            p = figure(sizing_mode = 'stretch_width', **figure_params)

        p = _core_create_scatterplot(fig_df, p, glyph_params, glyph_col = glyph_col, color_palette = color_palette, glyph_legend = glyph_legend, 
                                     glyph_legend_location= glyph_legend_location, glyph_legend_label_text_font_size=glyph_legend_label_text_font_size)

        if identity_line:
            p.add_layout(slope)

        fig = p
    else:  # Facet plot
        plots = []

        n = facet_max_subplot
        facet_col_items = fig_df[facet_col].unique().tolist()
        facet_col_items = sort_nicely(facet_col_items) if facet_sort_order else facet_col_items
        facet_col_items = [facet_col_items[i * n: (i + 1) * n] for i in range((len(facet_col_items) + n - 1) // n)]

        for i, fc_items in enumerate(facet_col_items):
            # Create plots
            fig = []
            linked_axes = {}
            for j, fc in enumerate(fc_items):
                p = figure(title=fc, **figure_params, **linked_axes)

                subset_facet = fig_df[fig_df[facet_col] == fc]

                p = _core_create_scatterplot(subset_facet, p, glyph_params, glyph_col = glyph_col, color_palette = color_palette, glyph_legend = glyph_legend, 
                                     glyph_legend_location= glyph_legend_location, glyph_legend_label_text_font_size=glyph_legend_label_text_font_size)

                if (j == 0) and (facet_sync_axes is not None):
                    if facet_sync_axes.lower() in ['x', 'both']:
                        linked_axes['x_range'] = p.x_range
                    if facet_sync_axes.lower() in ['y', 'both']:
                        linked_axes['y_range'] = p.y_range

                if identity_line:
                    p.add_layout(slope)

                fig.append(p)
            
            if height is None:
                fig = gridplot(fig, ncols=facet_col_wrap, sizing_mode='stretch_both', merge_tools=True)
            else:
                fig = gridplot(fig, ncols = facet_col_wrap, sizing_mode= 'stretch_width', merge_tools = True, height = height)

            if len(facet_col_items) > 1:  # If there will be multiple tabs, convert figure into a Panel
                start_num = i * n + 1
                end_num = i * n + len(fc_items)
                fig = Panel(child=fig, title=f'Plots {start_num}-{end_num}')

            plots.append(fig)

        if len(plots) == 1:
            fig = plots[0]
        else:
            fig = Tabs(tabs=plots)

    if figure_title is not None:
        fig = wrap_figure_title(fig, figure_title)

    if calc_pct_diff:
        df['pct_diff'] = (df[result_name] - df[controls_name]) / df[controls_name] * 100
        df['pct_diff'] = df['pct_diff'].replace([np.inf, -np.inf], np.nan)

    return df, fig
