import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Union, Hashable


def scatterplot_comparison(controls_df: pd.DataFrame, result_df: pd.DataFrame, ref_label: Union[str, List[str]],
                           data_label: str, category_labels: Dict = None, controls_name: str = 'controls',
                           result_name: str = 'model', figure_title: str = None, figure_height: int = None,
                           identity_line: bool = True, identity_colour: str = 'red', padding_value: int = 0,
                           **kwargs) -> Tuple[pd.DataFrame, go.Figure]:
    """Creates an interactive Plotly scatterplot to compare data.

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
        figure_title (str, optional): Defaults to ``None``. The chart title to use.
        figure_height (int, optional): Defaults to ``None``. The desired figure height.
        identity_line (bool, optional): Defaults to ``True``. A flag to include an identity (1:1) line in the
            scatterplot.
        identity_colour (str, optional): Defaults to ``'red'``. The colour to use for the identity line. Accepts html
            colour names.
        padding_value (int, optional): Defaults to ``0``. A padding value to set around the figure data when rendering.
        kwargs: Keyword/value pairs used to when creating Plotly figures.
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
    lower_val = -padding_value
    upper_val = max_val + padding_value
    axis_range = [lower_val, upper_val]

    # Plot figure
    fig = px.scatter(df, x=controls_name, y=result_name, hover_data=ref_label, range_x=axis_range, range_y=axis_range,
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


def tlfd_facet_plot(controls_df: pd.DataFrame, result_df: pd.DataFrame, data_label: str, bin_start: int = 0,
                    bin_end: int = 200, bin_step: int = 2, figure_title: str = None, figure_height: int = None,
                    controls_line_colour: str = 'red',  **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame, go.Figure]:
    """Create an interactive Plotly facet plot of TLFD diagrams using the trips table from a MicrosimData instance and a
    targets table.

    Args:
        controls_df (pd.DataFrame): A Pandas DataFrame containing TLFD targets in wide format (where rows correspond
            to the bins and columns correspond to the trips by category/group)
        result_df (pd.DataFrame): A Pandas DataFrame containing the model TLFDs in wide format (same ``controls_df``)
        data_label (str): The name to use for the data represented by the category/groups columns.
        bin_start (int): Defaults is ``0``. The minimum bin value.
        bin_end (int): Defaults to ``200``. The maximum bin value.
        bin_step (int): Default is ``2``. The size of each bin.
        figure_title (str, optional): Defaults to ``None``. The chart title to use.
        figure_height (int, optional): Defaults to ``None``. The desired figure height.
        controls_line_colour (str, optional): Defaults to ``'red'``. The colour to use for the control target lines.
            Accepts html colour names.
        kwargs: Keyword/value pairs used to when creating Plotly figures.

    Returns:
        Tuple[pd.DataFrame, go.Figure]: The TLFD distributions, and TLFD facet plots
    """

    assert set(controls_df) == set(result_df), 'The columns in `controls_df` and `result_df` do not match'
    # TODO: include comparison of indices

    # Calculate distributions
    model_tlfd_dist = result_df.div(result_df.sum(axis=0), axis=1)
    model_tlfd_dist.columns.name = data_label

    targets_tlfd_dist = controls_df.div(controls_df.sum(axis=0), axis=1)
    targets_tlfd_dist.columns.name = data_label

    df = _simplify_tlfd_index(model_tlfd_dist, low=bin_start - bin_step, high=bin_end).stack().to_frame(name='model')
    df['target'] = _simplify_tlfd_index(targets_tlfd_dist, low=bin_start - bin_step, high=bin_end).stack()
    df.reset_index(inplace=True)

    # Plot base figure
    facet_cols = 2
    fig = px.histogram(df, x='bin_start', y='model', nbins=len(model_tlfd_dist), facet_col=data_label,
                       facet_col_wrap=facet_cols, **kwargs)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))

    # Determine the row and column indicies for the subplots in the facet plot, to be used later to add target traces
    subplot_indices = {}
    for row_idx, row_figs in enumerate(fig._grid_ref):
        for col_idx, _ in enumerate(row_figs):
            sub_fig = fig.get_subplot(row=row_idx + 1, col=col_idx + 1)
            sub_fig_axes = tuple(sorted([sub_fig.xaxis.anchor, sub_fig.yaxis.anchor]))
            subplot_indices[sub_fig_axes] = (row_idx + 1, col_idx + 1)

    subplot_names = [a['text'] for a in fig.layout.annotations]

    filled_subplot_axes = [tuple(sorted([f.xaxis, f.yaxis])) for f in fig.data]
    filled_subplot_indices = [subplot_indices[axes] for axes in filled_subplot_axes]

    # Reorder lists to match subplot name to row and column indices (this should work as long as Plotly doesn't change
    # the ordering of subplot creation). subplot_names is extracted from figure annotations, which start from the bottom
    # and are ordered from left to right as you go up the facet rows. filled_subplot_indices is extracted based on the
    # axes of each subplot, which start at the top of the figure and are ordered from left to right as you go down the
    # facet rows. I wonder if there's a better way to deterime what data each subplot has...
    subplot_names = list(sum([tuple(subplot_names[max(n - facet_cols,0):n]) for n in range(len(subplot_names), 0, -facet_cols)], ()))
    filled_subplot_indices = list(sum([tuple(filled_subplot_indices[n:n + facet_cols]) for n in range(0, len(filled_subplot_indices), facet_cols)], ()))

    # Add targets as traces to base figure
    for name, indices in zip(subplot_names, filled_subplot_indices):
        row_idx, col_idx = indices
        subset_df = df[df[data_label] == name]
        target_line = go.Scatter(x=subset_df['bin_start'], y=subset_df['target'], mode='lines',
                                 line=go.scatter.Line(color=controls_line_colour), showlegend=False)
        fig.add_trace(target_line, row=row_idx, col=col_idx)

    # Final figure layout adjustments
    layout_args = dict()
    if figure_title is not None:
        layout_args['title_text'] = figure_title
    if figure_height is not None:
        layout_args['height'] = figure_height

    if len(layout_args) > 0:
        fig.update_layout(**layout_args)

    return df, fig
