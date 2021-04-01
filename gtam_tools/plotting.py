from bokeh.plotting import Figure, figure
from bokeh.models import CDSView, ColumnDataSource, Div, GroupFilter, Slope
from bokeh.layouts import Column, column, GridBox, gridplot
from bokeh.palettes import Category20
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Hashable
import warnings


def _prep_figure_params(x_label: str, y_label: str, tooltips: List[Tuple[str]], plot_height: int = None):
    figure_params = {
        'x_axis_label': x_label, 'y_axis_label': y_label, 'tooltips': tooltips, 'toolbar_location': 'above',
        'tools': 'pan,zoom_in,zoom_out,box_zoom,wheel_zoom,hover,save,reset', 'output_backend': 'webgl'
    }
    if plot_height is not None:
        figure_params['plot_height'] = plot_height

    return figure_params


def _wrap_figure_title(fig, figure_title: str):
    title = Div(text=f'<h2>{figure_title}</h2>')
    return column(children=[title, fig], sizing_mode='stretch_width')


def scatterplot_comparison(controls_df: pd.DataFrame, result_df: pd.DataFrame, data_label: str, *,
                           ref_label: Union[str, List[str]] = None, category_labels: Dict = None,
                           controls_name: str = 'controls', result_name: str = 'model', size: float = 7.5,
                           fill_alpha: float = 0.2, facet_col: str = None, facet_col_wrap: int = 2,
                           facet_sort_order: bool = True, facet_sync_axes: str = 'both',
                           hover_col: Union[str, List[str]] = None, glyph_col: str = None, glyph_legend: bool = True,
                           glyph_legend_location: str = 'bottom_right', glyph_legend_label_text_font_size: str = '11px',
                           figure_title: str = None, plot_height: int = None, identity_line: bool = True,
                           identity_colour: str = 'red', identity_width: int = 2
                           ) -> Tuple[pd.DataFrame, Union[Column, Figure, GridBox]]:
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
        plot_height (int, optional): Defaults to ``None``. The desired plot height. For facet plots, this value will be
            set for each subplot.
        identity_line (bool, optional): Defaults to ``True``. A flag to include an identity (1:1) line in the
            scatter plot.
        identity_colour (str, optional): Defaults to ``'red'``. The colour to use for the identity line. Accepts html
            colour names.
        identity_width (int, optional): Defaults to ``2``. The line width to use for the identity line.

    Returns:
        Tuple[pd.DataFrame, Union[Column, Figure, GridBox]]
    """

    if not controls_df.index.equals(result_df.index):
        warnings.warn('Indices for `controls_df` and `result_df` are not identical; function may not produce desired ' \
                      'results')
    if not controls_df.columns.equals(result_df.columns):
        warnings.warn('Columns for `controls_df` and `result_df` are not identical; function may not produce desired ' \
                      'results')

    if ref_label is None:
        assert np.all(controls_df.index.names == result_df.index.names), 'Unable to resolve different index names, ' \
                                                                         'please specify values for `ref_label` instead'
        assert not None in controls_df.index.names, 'Some index levels in `controls_df` do not have names'
        assert not None in result_df.index.names, 'Some index levels in `result_df` do not have names'
        ref_label = list(controls_df.index.names)
    elif isinstance(ref_label, Hashable):
        ref_label = [ref_label]
    elif isinstance(ref_label, List):
        pass
    else:
        raise RuntimeError('Invalid data type provided for `ref_label`')

    if hover_col is None:
        hover_col = []
    if isinstance(hover_col, Hashable):
        hover_col = [hover_col]
    elif isinstance(hover_col, List):
        pass
    else:
        raise RuntimeError('Invalid data type provided for `ref_label`')

    # Prepare data for plotting
    df = controls_df.stack()
    df.index.names = [*ref_label, data_label]
    df = df.to_frame(name=controls_name)

    df[result_name] = result_df.stack()
    df[result_name].fillna(0, inplace=True)
    df.reset_index(inplace=True)

    if category_labels is not None:
        df[data_label] = df[data_label].map(category_labels)

    color_palette = None
    if glyph_col is not None:
        assert len(df[glyph_col].unique()) <= 20, 'Number of colours in color palette exceeded (max 20)'
        color_palette = Category20[max(len(df[glyph_col].unique()), 3)]

    # Prepare figure formatting values
    source = ColumnDataSource(df)
    tooltips = [(c, '@{%s}' % c) for c in hover_col]
    tooltips += [(controls_name, '@{%s}{0,0.0}' % controls_name), (result_name, '@{%s}{0,0.0}' % result_name)]
    figure_params = _prep_figure_params(controls_name, result_name, tooltips, plot_height)
    glyph_params = {
        'source': source, 'x': controls_name, 'y': result_name, 'size': size, 'fill_alpha': fill_alpha,
        'hover_color': 'red'
    }

    slope = Slope(gradient=1, y_intercept=0, line_color=identity_colour, line_dash='dashed', line_width=identity_width)

    def apply_legend_settings(p_: Figure):
        p_.legend.visible = glyph_legend
        p_.legend.title = glyph_col
        p_.legend.location = glyph_legend_location
        p_.legend.label_text_font_size = glyph_legend_label_text_font_size
        p_.legend.click_policy = 'hide'

    # Plot figure
    if facet_col is None:  # Basic plot
        p = figure(sizing_mode='stretch_both', **figure_params)
        if glyph_col is None:  # Single glyphs
            p.circle(**glyph_params)
        else:  # Iterate through unique `glyph_col` values to use interactive legend feature
            for i, gc in enumerate(sorted(df[glyph_col].unique())):
                source_view = CDSView(source=source, filters=[GroupFilter(column_name=glyph_col, group=gc)])
                p.circle(view=source_view, legend_label=gc, color=color_palette[i], **glyph_params)
            apply_legend_settings(p)
        if identity_line:
            p.add_layout(slope)
        fig = p
    else:  # Facet plot
        fig = []
        facet_column_items = df[facet_col].unique().tolist()
        facet_column_items = sorted(facet_column_items) if facet_sort_order else facet_column_items
        linked_axes = {}
        for i, fc in enumerate(facet_column_items):
            p = figure(title=fc, **figure_params, **linked_axes)
            filters = [GroupFilter(column_name=facet_col, group=fc)]
            if glyph_col is None:  # Single glyphs
                source_view = CDSView(source=source, filters=filters)
                p.circle(view=source_view, **glyph_params)
            else:  # Iterate through unique `glyph_col` values to use interactive legend feature
                for j, gc in enumerate(sorted(df[glyph_col].unique())):
                    filters_ = filters + [GroupFilter(column_name=glyph_col, group=gc)]
                    source_view = CDSView(source=source, filters=filters_)
                    p.circle(view=source_view, legend_label=gc, color=color_palette[j], **glyph_params)
                apply_legend_settings(p)

            if (i==0) and (facet_sync_axes is not None):
                if facet_sync_axes.lower() in ['x', 'both']:
                    linked_axes['x_range'] = p.x_range
                if facet_sync_axes.lower() in ['y', 'both']:
                    linked_axes['y_range'] = p.y_range

            if identity_line:
                p.add_layout(slope)

            fig.append(p)
        fig = gridplot(fig, ncols=facet_col_wrap, sizing_mode='stretch_both', merge_tools=True)

    if figure_title is not None:
        fig = _wrap_figure_title(fig, figure_title)

    return df, fig


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


def tlfd_facet_plot(controls_df: pd.DataFrame, result_df: pd.DataFrame, data_label: str, *,
                    category_labels: Dict = None, bin_start: int = 0, bin_end: int = 200, bin_step: int = 2,
                    facet_col_wrap: int = 2, facet_sort_order: bool = True, facet_sync_axes: str = 'both',
                    legend_label_text_font_size: str = '11px', figure_title: str = None, plot_height: int = None,
                    controls_line_colour: str = 'red', controls_line_width: int = 2
                    ) -> Tuple[pd.DataFrame, Union[Column, GridBox]]:
    """Create an interactive Bokeh-based facet plot of TLFD diagrams using the trips table from a MicrosimData instance
    and a targets table.

    Args:
        controls_df (pd.DataFrame): A DataFrame containing TLFD targets in wide format (where rows correspond to the
            bins and columns correspond to the trips by category/group).
        result_df (pd.DataFrame): A DataFrame containing the model TLFDs in wide format (same ``controls_df``).
        data_label (str): The name to use for the data represented by the category/groups columns.
        category_labels (Dict, optional): Defaults to ``None``. Category labels used to rename the `controls_df` and
            `result_df` columns.
        bin_start (int): Defaults is ``0``. The minimum bin value.
        bin_end (int): Defaults to ``200``. The maximum bin value.
        bin_step (int): Default is ``2``. The size of each bin.
        facet_col_wrap (int, optional): Defaults to ``2``. The number of columns to wrap subplots in the facet plot.
        facet_sort_order (bool, optional): Defaults to ``True``. A flag to render facet subplots in ascending order
            sorted by unique ``facet_col`` values.
        facet_sync_axes (str, optional): Defaults to ``'both'``. Option to sync/link facet axes. Accepts one of
            ``['both', 'x', 'y']``. Set to None to disable linked facet plot axes.
        legend_label_text_font_size (str, optional): Defaults to ``'11px'``. The text size of the legend labels.
        figure_title (str, optional): Defaults to ``None``. The chart title to use.
        plot_height (int, optional): Defaults to ``None``. The desired plot height. For facet plots, this value will be
            set for each subplot.
        controls_line_colour (str, optional): Defaults to ``'red'``. The colour to use for the control target lines.
            Accepts html colour names.
        controls_line_width (int, optional): Defaults to ``2``. The line width to use for the control target lines.

    Returns:
        Tuple[pd.DataFrame, Union[Column, GridBox]]
    """

    if not controls_df.index.equals(result_df.index):
        warnings.warn('Indices for `controls_df` and `result_df` are not identical; function may not produce desired ' \
                      'results')
    if not controls_df.columns.equals(result_df.columns):
        warnings.warn('Columns for `controls_df` and `result_df` are not identical; function may not produce desired ' \
                      'results')

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

    # Prepare figure formatting values
    tooltips = [('bin_start', '@bin_start'), ('Model', '@model{0.3f}'), ('Target', '@target{0.3f}')]
    figure_params = _prep_figure_params('Bin', 'Proportion', tooltips, plot_height)

    # Plot figure
    fig = []
    facet_column_items = df[data_label].unique().tolist()
    facet_column_items = sorted(facet_column_items) if facet_sort_order else facet_column_items
    linked_axes = {}
    for i, fc in enumerate(facet_column_items):
        p = figure(title=fc, **figure_params, **linked_axes)
        subset = df[df[data_label] == fc]  # We do it this way because CDSView doesn't support connected lines
        source = ColumnDataSource(subset)
        p.line(source=source, x='bin_start', y='target', line_color=controls_line_colour,
               line_width=controls_line_width, legend_label='Target')
        p.vbar(source=source, x='bin_start', top='model', bottom=0, width=1.25, legent_label='Model')
        p.y_range.start = 0
        p.legend.label_text_font_size = legend_label_text_font_size

        if (i==0) and (facet_sync_axes is not None):
            if facet_sync_axes.lower() in ['x', 'both']:
                linked_axes['x_range'] = p.x_range
            if facet_sync_axes.lower() in ['y', 'both']:
                linked_axes['y_range'] = p.y_range

        fig.append(p)
    fig = gridplot(fig, ncols=facet_col_wrap, sizing_mode='stretch_both', merge_tools=True)

    if figure_title is not None:
        fig = _wrap_figure_title(fig, figure_title)

    return df, fig
