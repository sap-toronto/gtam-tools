import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Tuple


def scatterplot_comparison(controls_df: pd.DataFrame, result_df: pd.DataFrame, ref_label: str, data_label: str,
                           category_labels: Dict = None, controls_name: str = 'controls', result_name: str = 'model',
                           figure_title: str = None, figure_height: int = None, identity_line: bool = True,
                           identity_colour: str = 'red', padding_value: int = 0,
                           **kwargs) -> Tuple[pd.DataFrame, go.Figure]:
    """Creates an interactive Plotly scatterplot to compare data.

    Args:
        controls_df (pd.DataFrame): A DataFrame containing control values. Must be in wide-format where rows represent
            a reference (e.g. count station, TAZ, geography, etc.) and columns represent the data categories.
        result_df (pd.DataFrame): A DataFrame containing modelled values. Uses the same format as `controls_df`.
        ref_label (str): The reference variable name use.
        data_label (str): The data variable name to use.
        category_labels (Dict, optional): Defaults to ``None``. Category labels used to rename the `controls_df` and
            `result_df` columns.
        controls_name (str, optional): Defaults to ``'controls'``. The name for the controls.
        result_name (str, optional): Defaults to ``'model'``. The name for the results.
        figure_title (str, optional): Defaults to ``None``. The chart title to use.
        figure_height (int, optional): Defaults to ``None``. The desired figure height.
        identity_line (bool, optional): Defaults to ``True``. A flag to include an identity (1:1) line in the
            scatterplot.
        identity_colour (str, optional): Defaults to ``'red'``. The colour to use for the identity line. Accepts html
            colour names.
        padding_value (int, optional): Defaults to ``0``. A padding value to set around the figure data when rendering.
        kwargs: Keyword/value pairs used to when creating Plotly figures.
    """

    # Prepare data for plotting
    df = controls_df.stack()
    df.index.names = [ref_label, data_label]
    df = df.to_frame(name=controls_name)

    df[result_name] = result_df.stack()
    df[result_name].fillna(0, inplace=True)
    df.reset_index(inplace=True)

    if category_labels is not None:
        df[data_label] = df[data_label].map(category_labels)

    # Prepare figure formatting values
    max_val = int(df[[controls_name, result_name]].round(0).max().max())
    lower_val = -padding_value
    upper_val = max_val + padding_value
    axis_range = [lower_val, upper_val]

    # Plot figure
    fig = px.scatter(df, x=controls_name, y=result_name, hover_data=[ref_label], range_x=axis_range, range_y=axis_range,
                     **kwargs)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))

    layout_args = dict()
    if figure_title is not None:
        layout_args['title_text'] = figure_title
    if figure_height is not None:
        layout_args['height'] = figure_height

    if len(layout_args) > 0:
        fig.update_layout(**layout_args)

    if identity_line:
        ref_line = go.Scatter(x=axis_range, y=axis_range, mode='lines', line=go.scatter.Line(color=identity_colour),
                              showlegend=False)
        filled_subplots_axes = [set([f.xaxis, f.yaxis]) for f in fig.data]
        for row_idx, row_figs in enumerate(fig._grid_ref):
            for col_idx, _ in enumerate(row_figs):
                subplot_fig = fig.get_subplot(row=row_idx+1, col=col_idx+1)
                subplot_axes = set([subplot_fig.xaxis.anchor, subplot_fig.yaxis.anchor])
                if subplot_axes in filled_subplots_axes:
                    fig.add_trace(ref_line, row=row_idx+1, col=col_idx+1)

    return df, fig
