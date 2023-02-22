import warnings
from typing import Any, Dict, Hashable, List, Tuple, Union

import numpy as np
import pandas as pd
from balsa.routines import sort_nicely
from bokeh.layouts import Column, GridBox, column, gridplot
from bokeh.models import (BooleanFilter, CDSView, ColumnDataSource, Div,
                          FactorRange, GroupFilter, NumeralTickFormatter, Slope,
                          TabPanel, Tabs)
from bokeh.palettes import Category20, Set3
from bokeh.plotting import figure

# region Shared functions

def _check_df_indices(controls_df: pd.DataFrame, result_df: pd.DataFrame):
    if not controls_df.index.equals(result_df.index):
        warnings.warn('Indices for `controls_df` and `result_df` are not identical; function may not produce desired '
                      'results')
    if not controls_df.columns.equals(result_df.columns):
        warnings.warn('Columns for `controls_df` and `result_df` are not identical; function may not produce desired '
                      'results')


def _check_ref_label(ref_label: Union[str, List[str]], controls_df: pd.DataFrame, result_df: pd.DataFrame) -> List[str]:
    if ref_label is None:
        if not np.all(controls_df.index.names == result_df.index.names):
            raise RuntimeError('Unable to resolve different index names, please specify values for `ref_label` instead')
        if None in controls_df.index.names:
            raise RuntimeError('Some index levels in `controls_df` do not have names')
        if None in result_df.index.names:
            raise RuntimeError('Some index levels in `result_df` do not have names')
        ref_label = list(controls_df.index.names)
    elif isinstance(ref_label, Hashable):
        ref_label = [ref_label]
    elif isinstance(ref_label, List):
        pass
    else:
        raise RuntimeError('Invalid data type provided for `ref_label`')

    return ref_label


def _prep_figure_params(x_label: str, y_label: str, tooltips: List[Tuple[Hashable, Hashable]], plot_width: int = None,
                        plot_height: int = None) -> Dict[str, Any]:
    figure_params = {
        'x_axis_label': x_label, 'y_axis_label': y_label, 'tooltips': tooltips, 'toolbar_location': 'above',
        'tools': 'pan,zoom_in,zoom_out,box_zoom,wheel_zoom,hover,save,reset', 'output_backend': 'webgl'
    }
    if plot_width is not None:
        figure_params['width'] = plot_width
    if plot_height is not None:
        figure_params['height'] = plot_height

    return figure_params


def _wrap_figure_title(fig, figure_title: str) -> Column:
    title = Div(text=f'<h2>{figure_title}</h2>')
    return column(children=[title, fig], sizing_mode='stretch_width')

# endregion


# region Scatterplot comparison

def _prep_scatterplot_data(controls_df: pd.DataFrame, result_df: pd.DataFrame, data_label: str, ref_label: List[str], *,
                           category_labels: Dict = None, controls_name: str = 'controls', result_name: str = 'model',
                           totals_in_titles: bool = True, filter_zero_rows: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
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


def scatterplot_comparison(controls_df: pd.DataFrame, result_df: pd.DataFrame, data_label: str, *,
                           ref_label: Union[str, List[str]] = None, category_labels: Dict = None,
                           controls_name: str = 'controls', result_name: str = 'model', size: float = 7.5,
                           fill_alpha: float = 0.2, facet_col: str = None, facet_col_wrap: int = 2,
                           facet_sort_order: bool = True, facet_sync_axes: str = 'both', facet_max_subplot: int = 9,
                           hover_col: Union[str, List[str]] = None, glyph_col: str = None, glyph_legend: bool = True,
                           glyph_legend_location: str = 'bottom_right', glyph_legend_label_text_font_size: str = '11px',
                           figure_title: str = None, plot_width: int = None, plot_height: int = None,
                           sizing_mode: str = None, identity_line: bool = True, identity_colour: str = 'red',
                           identity_width: int = 2, color_palette: Dict[int, Any] = Category20,
                           calc_pct_diff: bool = True, totals_in_titles: bool = True, filter_zero_rows: bool = True
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
        plot_width (int, optional): Defaults to ``None``. The desired plot width. For facet plots, this value will be
            set for each subplot.
        plot_height (int, optional): Defaults to ``None``. The desired plot height. For facet plots, this value will be
            set for each subplot.
        sizing_mode (str, optional): Defaults to ``None``. A Bokeh SizingModeType. How will the items in the layout
            resize to fill the available space. Please refer to Bokeh documentation for acceptable values.
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
        Tuple[pd.DataFrame, Union[Column, Figure, GridBox, Tabs]]
    """
    _check_df_indices(controls_df, result_df)

    ref_label = _check_ref_label(ref_label, controls_df, result_df)

    if hover_col is None:
        hover_col = []
    if isinstance(hover_col, Hashable):
        hover_col = [hover_col]
    elif isinstance(hover_col, List):
        pass
    else:
        raise RuntimeError('Invalid data type provided for `ref_label`')

    df, fig_df = _prep_scatterplot_data(
        controls_df, result_df, data_label, ref_label, category_labels=category_labels, controls_name=controls_name,
        result_name=result_name, totals_in_titles=totals_in_titles, filter_zero_rows=filter_zero_rows
    )

    if glyph_col is not None:
        n_colors = max(len(fig_df[glyph_col].unique()), 3)
        color_palette = color_palette[n_colors]

    # Prepare figure formatting values
    source = ColumnDataSource(fig_df)
    tooltips = [(c, '@{%s}' % c) for c in hover_col]
    tooltips += [(controls_name, '@{%s}{0,0.0}' % controls_name), (result_name, '@{%s}{0,0.0}' % result_name)]
    figure_params = _prep_figure_params(
        controls_name, result_name, tooltips, plot_width=plot_width, plot_height=plot_height
    )
    glyph_params = {
        'source': source, 'x': controls_name, 'y': result_name, 'size': size, 'fill_alpha': fill_alpha,
        'hover_color': 'red'
    }

    slope = Slope(gradient=1, y_intercept=0, line_color=identity_colour, line_dash='dashed', line_width=identity_width)

    def apply_legend_settings(p_: figure):
        p_.legend.visible = glyph_legend
        p_.legend.title = glyph_col
        p_.legend.location = glyph_legend_location
        p_.legend.label_text_font_size = glyph_legend_label_text_font_size
        p_.legend.click_policy = 'hide'

    # Plot figure
    if facet_col is None:  # Basic plot
        p = figure(sizing_mode=sizing_mode, **figure_params)
        if glyph_col is None:  # Single glyphs
                p.circle(**glyph_params)
        else:  # Iterate through unique `glyph_col` values to use interactive legend feature
            for j, gc in enumerate(fig_df[glyph_col].unique()):
                glyph_group_filter = GroupFilter(column_name=glyph_col, group=gc)
                view = CDSView(filter=glyph_group_filter)
                p.circle(view=view, legend_label=gc, color=color_palette[j], **glyph_params)
            apply_legend_settings(p)
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
                p = figure(title=fc, sizing_mode=sizing_mode, **figure_params, **linked_axes)
                facet_group_filter = GroupFilter(column_name=facet_col, group=fc)
                if glyph_col is None:  # Single glyphs
                    view = CDSView(filter=facet_group_filter)
                    p.circle(view=view, **glyph_params)
                else:  # Iterate through unique `glyph_col` values to use interactive legend feature
                    for k, gc in enumerate(fig_df[glyph_col].unique()):
                        glyph_group_filter = GroupFilter(column_name=glyph_col, group=gc)
                        view = CDSView(filter=facet_group_filter & glyph_group_filter)
                        p.circle(view=view, legend_label=gc, color=color_palette[k], **glyph_params)
                    apply_legend_settings(p)

                if (j == 0) and (facet_sync_axes is not None):
                    if facet_sync_axes.lower() in ['x', 'both']:
                        linked_axes['x_range'] = p.x_range
                    if facet_sync_axes.lower() in ['y', 'both']:
                        linked_axes['y_range'] = p.y_range

                if identity_line:
                    p.add_layout(slope)

                fig.append(p)
            fig = gridplot(fig, ncols=facet_col_wrap, sizing_mode=sizing_mode, merge_tools=True)

            if len(facet_col_items) > 1:  # If there will be multiple tabs, convert figure into a TabPanel
                start_num = i * n + 1
                end_num = i * n + len(fc_items)
                fig = TabPanel(child=fig, title=f'Plots {start_num}-{end_num}')

            plots.append(fig)

        if len(plots) == 1:
            fig = plots[0]
        else:
            fig = Tabs(tabs=plots)

    if figure_title is not None:
        fig = _wrap_figure_title(fig, figure_title)

    if calc_pct_diff:
        df['pct_diff'] = (df[result_name] - df[controls_name]) / df[controls_name] * 100
        df['pct_diff'] = df['pct_diff'].replace([np.inf, -np.inf], np.nan)

    return df, fig

# endregion


# region Stacked horizontal-bar comparison

def _prep_stacked_hbar_data(controls_df: pd.DataFrame, result_df: pd.DataFrame, data_label: str, ref_label: List[str],
                            label_col: List[str], *, category_labels: Dict = None, controls_name: str = 'controls',
                            result_name: str = 'model', normalize: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = controls_df.copy()
    df.columns.name = data_label
    df = df.stack().to_frame(name=controls_name)
    df.index.names = [*ref_label, data_label]

    df[result_name] = result_df.stack()
    df = df.fillna(0).round(0).astype(np.int64)
    df.columns.name = 'source'
    df = df.stack().to_frame(name='total').reset_index()

    if category_labels is not None:
        df[data_label] = df[data_label].map(category_labels)

    # Prepare figure-specific data
    label_col_index = []
    for label in label_col:
        label_col_index.append(sort_nicely(df[label].unique().tolist()))  # perform a human sort
    label_col_index.append([controls_name, result_name])
    label_col_index = pd.MultiIndex.from_product(label_col_index, names=[*label_col, 'source'])

    fig_df = df.pivot_table(values='total', index=[*label_col, 'source'], columns=data_label, aggfunc='sum',
                            fill_value=0)
    mask = label_col_index.isin(fig_df.index)
    fig_df = fig_df.reindex(label_col_index[mask], fill_value=0)[::-1].copy()  # reindex and reverse
    fig_df.columns = fig_df.columns.astype(str)
    if normalize:
        fig_df = (fig_df / fig_df.sum(axis=1).to_numpy()[:, np.newaxis]).round(3)

    fig_df.reset_index(inplace=True)
    fig_df['label_col'] = [' - '.join([str(v) for v in l]) for l in fig_df[label_col].to_numpy().tolist()]
    fig_df.drop(label_col, axis=1, inplace=True)
    fig_df.set_index(['label_col', 'source'], inplace=True)

    return df, fig_df


def stacked_hbar_comparison(controls_df: pd.DataFrame, result_df: pd.DataFrame, data_label: str, *,
                            ref_label: Union[str, List[str]] = None, category_labels: Dict = None,
                            label_col: Union[str, List[str]] = None, controls_name: str = 'controls',
                            result_name: str = 'model', x_axis_label: str = None, figure_title: str = None,
                            plot_width: int = None, plot_height: int = None, sizing_mode: str = None,
                            normalize: bool = True, color_palette: Dict[int, Any] = Set3
                            ) -> Tuple[pd.DataFrame, Union[Column, figure, GridBox]]:
    """Creates an interactive Bokeh-based stacked horizontal bar chart to compare data

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
        label_col (Union[str, List[str]], optional): Defaults to ``None``. The columns to use when for figure axis
            grouping.
        controls_name (str, optional): Defaults to ``'controls'``. The name for the controls.
        result_name (str, optional): Defaults to ``'model'``. The name for the results.
        x_axis_label (str, optional): Defaults to ``None``. The label to apply to the x axis
        figure_title (str, optional): Defaults to ``None``. The chart title to use.
        plot_width (int, optional): Defaults to ``None``. The desired plot width. For facet plots, this value will be
            set for each subplot.
        plot_height (int, optional): Defaults to ``None``. The desired plot height. For facet plots, this value will be
            set for each subplot.
        sizing_mode (str, optional): Defaults to ``None``. A Bokeh SizingModeType. How will the items in the layout
            resize to fill the available space. Please refer to Bokeh documentation for acceptable values.
        normalize (bool, optional): Defaults to ``True``. Plot the stacked horizontal bar chart with normalized data.
        color_palette (Dict[str, Any], optional): Defaults to ``Set3``. The Bokeh color palette to use.

    Returns:
        Tuple[pd.DataFrame, Union[Column, Figure, GridBox]]
    """
    _check_df_indices(controls_df, result_df)

    ref_label = _check_ref_label(ref_label, controls_df, result_df)

    if label_col is None:
        label_col = controls_df.index.names
    elif isinstance(label_col, Hashable):
        label_col = [label_col]
    elif isinstance(label_col, List):
        pass
    else:
        raise RuntimeError('Invalid data type provided for `label_col`')

    # Prepare data
    df, fig_df = _prep_stacked_hbar_data(
        controls_df, result_df, data_label, ref_label, label_col, category_labels=category_labels,
        controls_name=controls_name, result_name=result_name, normalize=normalize
    )

    # Plot figure
    x_range = (0, 1) if normalize else (0, int(fig_df.sum(axis=1).max()))
    tooltips = '$name (@y) = @$name{0.0%}' if normalize else '$name (@y) = @$name'
    n_colors = max(len(df[data_label].unique()), 3)

    figure_params = {
        'toolbar_location': 'above', 'sizing_mode': sizing_mode, 'tools': 'xpan,xwheel_zoom,hover,save,reset',
        'output_backend': 'webgl'
    }
    if plot_width is not None:
        figure_params['width'] = plot_width
    if plot_height is not None:
        figure_params['height'] = plot_height
    if x_axis_label is not None:
        figure_params['x_axis_label'] = x_axis_label

    source = fig_df.to_dict(orient='list')
    source['y'] = fig_df.index.tolist()
    source = ColumnDataSource(source)

    factors = fig_df.index.tolist()
    columns = fig_df.columns.tolist()

    fig = figure(x_range=x_range, y_range=FactorRange(*factors), tooltips=tooltips, **figure_params)

    fig.hbar_stack(columns, y='y', source=source, color=color_palette[n_colors], legend_label=columns)

    if normalize:
        fig.xaxis.formatter = NumeralTickFormatter(format='0%')

    fig.y_range.factor_padding = 0.25
    fig.y_range.group_padding = 1

    fig.yaxis.group_label_orientation = 'horizontal'

    fig.add_layout(fig.legend[0], 'above')
    fig.legend.margin = 20
    fig.legend.orientation = 'horizontal'
    fig.legend.spacing = 5

    if figure_title is not None:
        fig = _wrap_figure_title(fig, figure_title)

    return df, fig

# endregion


# region TLFD facet plot

def _simplify_tlfd_index(df: Union[pd.DataFrame, pd.Series], low: float = -2.0,
                         high: float = 200.0) -> Union[pd.DataFrame, pd.Series]:
    new_df = df.copy()

    new_index = [float(low)]
    for f, _ in new_df.index.values[1:-1]:
        new_index.append(float(f))
    new_index.append(float(high))
    new_df.index = new_index
    new_df.index.name = 'bin_start'

    return new_df


def _prep_tlfd_data(controls_df: pd.DataFrame, result_df: pd.DataFrame, data_label: str, *,
                    category_labels: Dict = None, bin_start: int = 0, bin_end: int = 200,
                    bin_step: int = 2) -> pd.DataFrame:
    # Calculate distributions
    model_tlfd_dist = result_df.div(result_df.sum(axis=0), axis=1)
    model_tlfd_dist.columns.name = data_label

    targets_tlfd_dist = controls_df.div(controls_df.sum(axis=0), axis=1)
    targets_tlfd_dist.columns.name = data_label

    df = _simplify_tlfd_index(model_tlfd_dist, low=bin_start - bin_step, high=bin_end).stack().to_frame(name='model')
    df['target'] = _simplify_tlfd_index(targets_tlfd_dist, low=bin_start - bin_step, high=bin_end).stack()
    df.reset_index(inplace=True)

    if category_labels is not None:
        df[data_label] = df[data_label].map(category_labels)

    return df


def tlfd_facet_plot(controls_df: pd.DataFrame, result_df: pd.DataFrame, data_label: str, *,
                    category_labels: Dict = None, controls_name: str = 'controls', result_name: str = 'model',
                    bin_start: int = 0, bin_end: int = 200, bin_step: int = 2, facet_col_wrap: int = 2,
                    facet_sort_order: bool = True, facet_sync_axes: str = 'both', facet_max_subplot: int = 9,
                    legend_label_text_font_size: str = '11px', figure_title: str = None, plot_width: int = None,
                    plot_height: int = None, sizing_mode: str = None, controls_line_colour: str = 'red',
                    controls_line_width: int = 2) -> Tuple[pd.DataFrame, Union[Column, GridBox, Tabs]]:
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
        facet_sync_axes (str, optional): Defaults to ``'both'``. Option to sync/link facet axes. Accepts one of
            ``['both', 'x', 'y']``. Set to None to disable linked facet plot axes.
        facet_max_subplot (int, optional): Defaults to ``9``. The maximum number of facet subplots per tab. If the
            number of subplots exceed this value, a tabbed interface will be used.
        legend_label_text_font_size (str, optional): Defaults to ``'11px'``. The text size of the legend labels.
        figure_title (str, optional): Defaults to ``None``. The chart title to use.
        plot_width (int, optional): Defaults to ``None``. The desired plot width. For facet plots, this value will be
            set for each subplot.
        plot_height (int, optional): Defaults to ``None``. The desired plot height. For facet plots, this value will be
            set for each subplot.
        sizing_mode (str, optional): Defaults to ``None``. A Bokeh SizingModeType. How will the items in the layout
            resize to fill the available space. Please refer to Bokeh documentation for acceptable values.
        controls_line_colour (str, optional): Defaults to ``'red'``. The colour to use for the control target lines.
            Accepts html colour names.
        controls_line_width (int, optional): Defaults to ``2``. The line width to use for the control target lines.

    Returns:
        Tuple[pd.DataFrame, Union[Column, GridBox]]
    """
    _check_df_indices(controls_df, result_df)

    df = _prep_tlfd_data(
        controls_df, result_df, data_label, category_labels=category_labels, bin_start=bin_start, bin_end=bin_end,
        bin_step=bin_step
    )

    # Prepare figure formatting values
    tooltips = [('bin_start', '@bin_start'), (result_name, '@model{0.3f}'), (controls_name, '@target{0.3f}')]
    figure_params = _prep_figure_params('Bin', 'Proportion', tooltips, plot_width=plot_width, plot_height=plot_height)

    # Plot figure
    plots = []

    n = facet_max_subplot
    facet_col_items = df[data_label].unique().tolist()
    facet_col_items = sort_nicely(facet_col_items) if facet_sort_order else facet_col_items
    facet_col_items = [facet_col_items[i * n: (i + 1) * n] for i in range((len(facet_col_items) + n - 1) // n)]

    for i, fc_items in enumerate(facet_col_items):
        # Create plots
        fig = []
        linked_axes = {}
        for j, fc in enumerate(fc_items):
            p = figure(title=fc, **figure_params, **linked_axes)
            subset = df[df[data_label] == fc]  # We do it this way because CDSView doesn't support connected lines
            source = ColumnDataSource(subset)
            p.line(source=source, x='bin_start', y='target', line_color=controls_line_colour,
                   line_width=controls_line_width, legend_label=controls_name)
            p.vbar(source=source, x='bin_start', top='model', bottom=0, width=1.25, legend_label=result_name,
                   fill_alpha=0.2)
            p.y_range.start = 0
            p.legend.label_text_font_size = legend_label_text_font_size

            if (j == 0) and (facet_sync_axes is not None):
                if facet_sync_axes.lower() in ['x', 'both']:
                    linked_axes['x_range'] = p.x_range
                if facet_sync_axes.lower() in ['y', 'both']:
                    linked_axes['y_range'] = p.y_range

            fig.append(p)
        fig = gridplot(fig, ncols=facet_col_wrap, sizing_mode=sizing_mode, merge_tools=True)

        if len(facet_col_items) > 1:  # If there will be multiple tabs, convert figure into a TabPanel
            start_num = i * n + 1
            end_num = i * n + len(fc_items)
            fig = TabPanel(child=fig, title=f'Plots {start_num}-{end_num}')

        plots.append(fig)

    if len(plots) == 1:
        fig = plots[0]
    else:
        fig = Tabs(tabs=plots)

    if figure_title is not None:
        fig = _wrap_figure_title(fig, figure_title)

    return df, fig

# endregion
