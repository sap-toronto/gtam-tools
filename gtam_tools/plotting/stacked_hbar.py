from __future__ import annotations

from typing import Any, Dict, Hashable, List, Tuple, Union

import numpy as np
import pandas as pd

from balsa.routines import sort_nicely
from bokeh.models import ColumnDataSource, FactorRange, NumeralTickFormatter
from bokeh.layouts import Column, GridBox
from bokeh.palettes import Category20, Category10, Category20b, Category20c, Set1, Set2, Set3
from bokeh.plotting import figure

from common import check_df_indices, check_ref_label, wrap_figure_title


def prep_stacked_hbar_data(controls_df: pd.DataFrame, result_df: pd.DataFrame, data_label: str, ref_label: Union[str, List[str]],
                           label_col: Union[str, List[str]], *, category_labels: Dict = None, controls_name: str = 'controls',
                           result_name: str = 'model', normalize: bool = True)-> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepares the data for plotting

    Args:
        controls_df (pd.DataFrame): A DataFrame containing control values. Must be in wide-format where rows represent a reference and columns represent data categories.
        result_df (pd.DataFrame): A DataFrame containing modelled values. Must be in the same format as 'controls_df'.
        data_label (str): The name to use for the data represented by the 'controls_df' and 'result_df' columns.
        ref_label (Union[str, List[str]]): The name(s) corresponding to the 'controls_df' and 'result_df' indices.
        label_col (Union[str, List[str]]): The column(s) to use for figure axis grouping. 
        category_labels (Dict, optional): Defaults to 'None'. Used to rename the 'controls_df' and 'result_df' columns.
        controls_name (str, optional): Defaults to 'controls'. The name for the controls.
        result_name (str, optional): Defaults to 'model'. The name for the results.
        normalize (bool, optional):  Defaults to 'True'. Plot the stacked horizontal bar chart with normalized data. 

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]
    """
    # Prepare data
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
                            ref_label: Union[str, List[str]] = None,  category_labels: Dict = None,
                            label_col: Union[str, List[str]] = None, controls_name: str = 'controls',
                            result_name: str = 'model', x_axis_label: str = None, figure_title: str = None,
                            height: int = None, normalize: bool = True, color_palette: Dict[int, Any] = Set3
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
        label_col (Union[str, List[str]], optional): Defaults to ``None``. The columns to use for figure axis
            grouping.
        controls_name (str, optional): Defaults to ``'controls'``. The name for the controls.
        result_name (str, optional): Defaults to ``'model'``. The name for the results.
        x_axis_label (str, optional): Defaults to ``None``. The label to apply to the x axis
        figure_title (str, optional): Defaults to ``None``. The chart title to use.
        height (int, optional): Defaults to ``None``. The desired plot height. For facet plots, this value will be
            set for each subplot.
        normalize (bool, optional): Defaults to ``True``. Plot the stacked horizontal bar chart with normalized data.
        color_palette (Dict[str, Any], optional): Defaults to ``Set3``. The Bokeh color palette to use.

    Returns:
        Tuple[pd.DataFrame, Union[Column, Figure, GridBox]]
    """
    check_df_indices(controls_df, result_df)

    ref_label = check_ref_label(ref_label, controls_df, result_df)
    if label_col is None:
        label_col = controls_df.index.names
    elif isinstance(label_col, Hashable):
        label_col = [label_col]
    elif isinstance(label_col, List):
        pass
    else:
        raise RuntimeError('Invalid data type provided for `label_col`')

    df, fig_df = prep_stacked_hbar_data(controls_df, result_df, data_label, ref_label, label_col, category_labels = category_labels, 
                                        controls_name = controls_name, result_name = result_name, normalize = normalize)

    # Plot figure
    x_range = (0, 1) if normalize else (0, int(fig_df.sum(axis=1).max()))
    tooltips = '$name (@y) = @$name{0.0%}' if normalize else '$name (@y) = @$name'
    n_colors = max(len(df[data_label].unique()), 3)

    figure_params = {
        'toolbar_location': 'above', 'tools': 'xpan,xwheel_zoom,hover,save,reset',
        'output_backend': 'webgl'
    }

    if x_axis_label is not None:
        figure_params['x_axis_label'] = x_axis_label

    source = fig_df.to_dict(orient='list')
    source['y'] = fig_df.index.tolist()
    source = ColumnDataSource(source)

    factors = fig_df.index.tolist()
    columns = fig_df.columns.tolist()

    if height is not None:
        figure_params['height'] = height
        fig = figure(x_range=x_range, y_range=FactorRange(*factors), tooltips=tooltips, **figure_params, sizing_mode = 'stretch_width')
    else:
        fig = figure(x_range=x_range, y_range=FactorRange(*factors), tooltips=tooltips, **figure_params, sizing_mode = 'stretch_both')

    try:
        fig.hbar_stack(columns, y='y', source=source, color=color_palette[n_colors], legend_label=columns)
    except KeyError as err:
        raise RuntimeError('Color Palette does not contain sufficient unique colors for your data.')

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
        fig = wrap_figure_title(fig, figure_title)

    return df, fig