from bokeh.plotting import Figure, figure
from bokeh.models import CDSView, ColumnDataSource, Div, GroupFilter
from bokeh.layouts import Column, column, GridBox, gridplot
from bokeh.palettes import Category20
import pandas as pd
from typing import Dict, List, Tuple, Union, Hashable

_BOKEH_PLOT_TOOLS = 'pan,zoom_in,zoom_out,box_zoom,wheel_zoom,hover,save,reset'


def scatterplot_comparison(controls_df: pd.DataFrame, result_df: pd.DataFrame, ref_label: Union[str, List[str]],
                           data_label: str, *, category_labels: Dict = None, controls_name: str = 'controls',
                           result_name: str = 'model', size: float = 7.5, fill_alpha: float = 0.2,
                           facet_col: str = None, facet_col_wrap: int = 2, sort_facet_order: bool = True,
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
        ref_label (Union[str, List[str]]): The reference variable name(s) use. A list of names is required if
            ``controls_df`` and ``result_df`` indices are MultiIndex objects
        data_label (str): The data variable name to use.
        category_labels (Dict, optional): Defaults to ``None``. Category labels used to rename the `controls_df` and
            `result_df` columns.
        controls_name (str, optional): Defaults to ``'controls'``. The name for the controls.
        result_name (str, optional): Defaults to ``'model'``. The name for the results.
        size (float, optional): Defaults to ``7.5``. The size of the scatter plot points.
        fill_alpha (float, optional): Defaults to ``0.2``. The opacity of the point fill.
        facet_col (str, optional): Defaults to ``None``. The name of the column to use for creating a facet plot.
        facet_col_wrap (int, optional): Defaults to ``2``. The number of columns to wrap subplots in the facet plot.
        sort_facet_order (bool, optional): Defaults to ``True``. A flag to render facet subplots in ascending order
            sorted by unique ``facet_col`` values.
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

    if isinstance(ref_label, Hashable):
        ref_label = [ref_label]
    elif isinstance(ref_label, List):
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

    # Prepare figure formatting values
    max_val = int(df[[controls_name, result_name]].round(0).max().max())
    source = ColumnDataSource(df)
    tooltips = [(controls_name, f'@{controls_name}{{0,0.0}}'), (result_name, f'@{result_name}{{0,0.0}}')]
    if hover_col is not None:
        if isinstance(hover_col, Hashable):
            hover_col = [hover_col]
        elif isinstance(hover_col, List):
            pass
        else:
            raise RuntimeError('Invalid data type provided for `ref_label`')
        for c in reversed(hover_col):
            tooltips.insert(0, (c, f'@{c}'))
    figure_params = {
        'x_axis_label': controls_name, 'y_axis_label': result_name, 'tooltips': tooltips, 'tools': _BOKEH_PLOT_TOOLS,
        'output_backend': 'webgl'
    }
    glyph_params = {
        'x': controls_name, 'y': result_name, 'size': size, 'fill_alpha': fill_alpha, 'hover_color': 'red'
    }

    def add_identity_line(p_: Figure):
        p_.line([0, max_val], [0, max_val], color=identity_colour, line_width=identity_width)

    def apply_legend_settings(p_: Figure):
        p_.legend.visible = glyph_legend
        p_.legend.title = glyph_col
        p_.legend.location = glyph_legend_location
        p_.legend.label_text_font_size = glyph_legend_label_text_font_size
        p_.legend.click_policy = 'hide'

    # Plot figure
    if facet_col is None:  # Basic plot
        p = figure(plot_height=plot_height, sizing_mode='stretch_width', **figure_params)
        if identity_line:
            add_identity_line(p)

        if glyph_col is None:  # Basic plot
            p.circle(source=source, **glyph_params)
        else:  # Iterate through unique `glyph_col` values to use interactive legend feature
            color_palette = Category20[len(df[glyph_col].unique())]
            for i, gc in enumerate(sorted(df[glyph_col].unique())):
                source_view = CDSView(source=source, filters=[GroupFilter(column_name=glyph_col, group=gc)])
                p.circle(source=source, view=source_view, legend_label=gc, color=color_palette[i], **glyph_params)
            apply_legend_settings(p)
        fig = p
    else:  # Facet plot
        fig = []
        facet_column_items = df[facet_col].unique().tolist()
        facet_column_items = sorted(facet_column_items) if sort_facet_order else facet_column_items
        for fc in facet_column_items:
            p = figure(title=fc, **figure_params)
            if identity_line:
                add_identity_line(p)

            filters = [GroupFilter(column_name=facet_col, group=fc)]
            if glyph_col is None:  # Basic facet plot
                source_view = CDSView(source=source, filters=filters)
                p.circle(source=source, view=source_view, **glyph_params)
            else:  # Iterate through unique `glyph_col` values to use interactive legend feature
                color_palette = Category20[len(df[glyph_col].unique())]
                for i, gc in enumerate(sorted(df[glyph_col].unique())):
                    filters_ = filters + [GroupFilter(column_name=glyph_col, group=gc)]
                    source_view = CDSView(source=source, filters=filters_)
                    p.circle(source=source, view=source_view, legend_label=gc, color=color_palette[i], **glyph_params)
                apply_legend_settings(p)
            fig.append(p)
        fig = gridplot(fig, ncols=facet_col_wrap, sizing_mode='stretch_both', plot_height=plot_height, merge_tools=True)

    if figure_title is not None:
        title = Div(text=f'<h2>{figure_title}</h2>')
        fig = column(children=[title, fig], sizing_mode='stretch_width')

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
                    facet_col_wrap: int = 2, sort_facet_order: bool = True, figure_title: str = None,
                    plot_height: int = None, controls_line_colour: str = 'red', controls_line_width: int = 2
                    ) -> Tuple[pd.DataFrame, Union[Column, GridBox]]:
    """Create an interactive Bokeh-based facet plot of TLFD diagrams using the trips table from a MicrosimData instance
    and a targets table.

    Args:
        controls_df (pd.DataFrame): A Pandas DataFrame containing TLFD targets in wide format (where rows correspond
            to the bins and columns correspond to the trips by category/group)
        result_df (pd.DataFrame): A Pandas DataFrame containing the model TLFDs in wide format (same ``controls_df``)
        data_label (str): The name to use for the data represented by the category/groups columns.
        category_labels (Dict, optional): Defaults to ``None``. Category labels used to rename the `controls_df` and
            `result_df` columns.
        bin_start (int): Defaults is ``0``. The minimum bin value.
        bin_end (int): Defaults to ``200``. The maximum bin value.
        bin_step (int): Default is ``2``. The size of each bin.
        facet_col_wrap (int, optional): Defaults to ``2``. The number of columns to wrap subplots in the facet plot.
        sort_facet_order (bool, optional): Defaults to ``True``. A flag to render facet subplots in ascending order
            sorted by unique ``facet_col`` values.
        figure_title (str, optional): Defaults to ``None``. The chart title to use.
        plot_height (int, optional): Defaults to ``None``. The desired plot height. For facet plots, this value will be
            set for each subplot.
        controls_line_colour (str, optional): Defaults to ``'red'``. The colour to use for the control target lines.
            Accepts html colour names.
        controls_line_width (int, optional): Defaults to ``2``. The line width to use for the control target lines.

    Returns:
        Tuple[pd.DataFrame, Union[Column, GridBox]]
    """

    assert set(controls_df) == set(result_df), 'The columns in `controls_df` and `result_df` do not match'

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
    tooltips = [('bin_start', '@bin_start'), ('model', '@model{0.3f}'), ('target', '@target{0.3f}')]
    figure_params = {
        'x_axis_label': 'Bin', 'y_axis_label': 'Proportion', 'tooltips': tooltips, 'tools': _BOKEH_PLOT_TOOLS,
        'output_backend': 'webgl'
    }

    # Plot figure
    fig = []
    facet_column_items = df[data_label].unique().tolist()
    facet_column_items = sorted(facet_column_items) if sort_facet_order else facet_column_items
    for fc in facet_column_items:
        p = figure(title=fc, **figure_params)
        subset = df[df[data_label] == fc]  # We do it this way because CDSView doesn't support connected lines
        source = ColumnDataSource(subset)
        p.line(x='bin_start', y='target', line_color=controls_line_colour, line_width=controls_line_width,
               source=source)
        p.vbar(x='bin_start', top='model', bottom=0, source=source)
        p.x_range.range_padding = 0.1
        fig.append(p)
    fig = gridplot(fig, ncols=facet_col_wrap, sizing_mode='stretch_both', plot_height=plot_height, merge_tools=True)

    if figure_title is not None:
        title = Div(text=f'<h2>{figure_title}</h2>')
        fig = column(children=[title, fig], sizing_mode='stretch_width')

    return df, fig
