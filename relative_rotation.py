"""Relative Rotation Studies"""

import asyncio
import warnings
from datetime import (
    date as dateType,
    datetime,
    timedelta,
)
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
from openbb_charting.core.openbb_figure import OpenBBFigure
from openbb_core.app.command_runner import CommandRunner
from openbb_core.app.model.charts.chart import Chart, ChartFormat
from openbb_core.app.model.obbject import OBBject
from openbb_core.app.utils import basemodel_to_df, df_to_basemodel
from openbb_core.provider.abstract.data import Data
from pandas import DataFrame, Series, to_datetime
from plotly import graph_objects as go

_warn = warnings.warn


color_sequence = [
    "burlywood", "orange", "grey", "magenta", "cyan", "yellowgreen",
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
    "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5",
    "#7e7e7e", "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#666666", "#f0027f",
    "#bf5b17", "#6666a", "#d9f202", "#d9f02", "#7573b", "#e798a", "#66a16", "#e6a02", "#a671d", "#66666",
    "#f007f", "#bf517", "#6666a", "#d9f202", "#d9f02", "#7573b", "#e798a", "#66a16", "#e6a02", "#a671d"
]


async def get_data(
    symbols: List[str],
    benchmark: str,
    study: Literal["price", "volume", "volatility"] = "price",
    date: Optional[dateType] = None,
    tail_periods: Optional[int] = 16,
    tail_interval: Literal["week", "month"] = "week",
    provider: str = "yfinance",
) -> Tuple[DataFrame, DataFrame]:
    """
    Fetch data with the OpenBB Platform Command Runner.

    Parameters
    ----------
    symbols: List[str]
        List of symbols to fetch data for.
    benchmark: str
        Benchmark ticker symbol to compare against.
    study: Literal["price", "volume", "volatility"]
        The type of data to study. Default is "price".
    date: Optional[dateType]
        The date representing the end date of the study.  Default is today.
    provider: str
        The OpenBB Platform data provider to use. Default is yfinance.
    """
    backfill = 156
    if tail_interval == "week" and tail_periods > backfill:
        backfill = tail_periods
    if tail_interval == "month" and tail_periods * 4 > backfill:
        backfill = tail_periods * 4

    if date is not None:
        date = to_datetime(date)
        start_date = (date - timedelta(weeks=backfill)).date()
        end_date = date
    if date is None:
        start_date = (datetime.now() - timedelta(weeks=backfill)).date()
        end_date = datetime.now().date()

    tasks = [
        CommandRunner().run(
            "/equity/price/historical",
            provider_choices={
                "provider": provider,
            },
            standard_params={
                "symbol" : ",".join(symbols),
                "start_date": start_date,
                "end_date": end_date,
                "interval": "1d",
            },
            extra_params={"use_cache": False}
        ),  # type: ignore
        CommandRunner().run(
            "/equity/price/historical",
            provider_choices={
                "provider": provider,
            },
            standard_params={
                "symbol" : benchmark,
                "start_date": start_date,
                "end_date": end_date,
                "interval": "1d",
            },
            extra_params={"use_cache": False}
        ),  # type: ignore
    ]
    target_column = "volume" if study == "volume" else "close"
    symbols_df, benchmark_df = await asyncio.gather(*tasks)
    try:
        symbols_data = symbols_df.to_df()
    except Exception as e:
        raise RuntimeError(f"There was an error loading data for {symbols}: {e}")
    tickers = symbols_data["symbol"].unique().tolist() if "symbol" in symbols_data.columns else symbols
    prices_data = (
        symbols_data.pivot(columns="symbol", values=target_column)
        if len(tickers) > 1
        else symbols_data[target_column].to_frame()
    )
    try:
        benchmark_data = benchmark_df.to_df()
    except Exception as e:
        raise RuntimeError(
            f"There was an error loading data for {benchmark}: {e}."
            " Check if the ticker symbol is valid for the provider."
        )
    bench_target = (
        "volume" if target_column == "volume"
        and "volume" in symbols_data.columns
        else "close"
    )
    if target_column == "volume" and bench_target == "close":
        _warn(
            "Volume data not available for benchmark. Using close price."
            "To study volume against the benchmark, use an index-tracking ETF."
        )

    benchmark_data = benchmark_data[[bench_target]].rename(columns={bench_target: benchmark})
    if len(benchmark_data) != len(prices_data):
        raise RuntimeError(
            "Benchmark data and symbols data must be the same length."
            " Check completeness for all symbols and benchmark "
            "by using `obb.equity.price.historical()`"
        )
    return prices_data, benchmark_data


def absolute_maximum_scale(data: Series) -> Series:
    """Absolute Maximum Scale Normaliztion Method"""
    return data / data.abs().max()


def min_max_scaling(data: Series) -> Series:
    """"Min/Max ScalingNormalization Method"""
    return (data - data.min()) / (data.max() - data.min())


def z_score_standardization(data:  Series) -> Series:
    """Z-Score Standardization Method."""
    return (data - data.mean()) / data.std()


def normalize(data: DataFrame, method: Literal["z", "m", "a"] = "z") -> DataFrame:
    """
    Normalize a Pandas DataFrame based on method.

    Parameters
    -----------
    data: DataFrame
        Pandas DataFrame with any number of columns to be normalized.
    method: Literal["z", "m", "a"]
        Normalization method.
            z: Z-Score Standardization
            m: Min/Max Scaling
            a: Absolute Maximum Scale

    Returns
    --------
    DataFrame
        Normalized DataFrame.
    """

    methods = {"z": z_score_standardization, "m": min_max_scaling, "a": absolute_maximum_scale}

    df = data.copy()

    for col in df.columns:
        df.loc[:, col] = methods[f"{method}"](df.loc[:, col])

    return df


def standard_deviation(
    data: DataFrame,
    window: int = 21,
    trading_periods: Optional[int] = None,
    is_crypto: bool = False,
    clean: bool = True,
) -> DataFrame:
    """
    Standard deviation.

    Measures how widely returns are dispersed from the average return.
    It is the most common (and biased) estimator of volatility.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe of OHLC prices.
    window : int [default: 21]
        Length of window to calculate over.
    trading_periods : Optional[int] [default: 252]
        Number of trading periods in a year.
    is_crypto : bool [default: False]
        If true, trading_periods is defined as 365.
    clean : bool [default: True]
        Whether to clean the data or not by dropping NaN values.

    Returns
    -------
    pd.DataFrame : results
        Dataframe with results.
    """
    data = data.copy()
    results = DataFrame()
    if window < 2:
        _warn("Error: Window must be at least 2, defaulting to 21.")
        window = 21

    if trading_periods and is_crypto:
        _warn("is_crypto is overridden by trading_periods.")

    if not trading_periods:
        trading_periods = 365 if is_crypto else 252

    for col in data.columns.tolist():
        log_return = (data[col] / data[col].shift(1)).apply(np.log)

        result = log_return.rolling(window=window, center=False).std() * np.sqrt(
            trading_periods
        )
        results[col] = result

    if clean:
        return results.dropna()

    return results


def calculate_momentum(data: Series, long_period: int = 252, short_period: int = 21) -> Series:
    """
    Momentum is calculated as the log trailing 12-month return minus trailing one-month return.
    Higher values indicate larger, positive momentum exposure.

    Momentum = ln(1 + r12) - ln(1 + r1)

    Parameters
    ----------
    data: Series
        Time series data to calculate the momentum for.
    long_period: Optional[int]
        Long period to base the calculation on. Default is one standard trading year.
    short_period: Optional[int]
        Short period to subtract from the long period. Default is one trading month.

    Returns
    -------
    Series
        Pandas Series with the calculated momentum.
    """
    df = data.copy()
    epsilon = 1e-10
    momentum_long = np.log(1 + df.pct_change(long_period) + epsilon)
    momentum_short = np.log(1 + df.pct_change(short_period) + epsilon)
    data = momentum_long - momentum_short  # type: ignore

    return data

def get_momentum(data: DataFrame,  long_period: int = 252, short_period: int = 21) -> DataFrame:
    """
    Calculate the Relative-Strength Momentum Indicator.  Takes the Relative Strength Ratio as the input.

    Parameters
    ----------
    data: DataFrame
        Indexed time series data formatted with each column representing a ticker.
    long_period: Optional[int]
        Long period to base the calculation on. Default is one standard trading year.
    short_period: Optional[int]
        Short period to subtract from the long period. Default is one trading month.

    Returns
    -------
    DataFrame
        Pandas DataFrame with the calculated historical momentum factor exposure score.
    """

    df = data.copy()
    rs_momentum = DataFrame()
    for ticker in df.columns.to_list():
        rs_momentum.loc[:, ticker] = calculate_momentum(df.loc[:, ticker], long_period, short_period)  # type: ignore

    return rs_momentum


def calculate_relative_strength_ratio(
    symbols_data: DataFrame,
    benchmark_data: DataFrame,
) -> DataFrame:
    """
    Calculate the Relative Strength Ratio for each ticker (column) in a DataFrame against the benchmark.
    symbols data and benchmark data should have the same index, and each column should represent a ticker.

    Parameters
    -----------
    symbols_data: DataFrame
        Pandas DataFrame with the symbols data to compare against the benchmark.
    benchmark_data: DataFrame
        Pandas DataFrame with the benchmark data.

    Returns
    --------
    DataFrame
        Pandas DataFrame with the calculated relative strength ratio for each ticker joined with the benchmark values.
    """
    return (
        symbols_data.div(benchmark_data.iloc[:, 0], axis=0)
        .multiply(100)
        .join(benchmark_data.iloc[:, 0])
        .dropna()
    )


def process_data(
    symbols_data: DataFrame,
    benchmark_data: DataFrame,
    long_period: int = 252,
    short_period: int = 21,
    normalize_method: Literal["z", "m", "a"] = "z"
) -> Tuple[DataFrame, DataFrame]:
    """
    Process the raw data into normalized indicator values.

    Parameters
    ----------
    symbols_data: DataFrame
        Indexed time series data formatted with each column representing a ticker.
    benchmark_data: DataFrame
        Indexed time series data of the benchmark symbol.
    long_period: Optional[int]
        Long period to base the calculation on. Default is one standard trading year.
    short_period: Optional[int]
        Short period to subtract from the long period. Default is one trading month.
    normalize_method: Literal["z", "m", "a"]

    Returns
    -------
    Tuple[DataFrame, DataFrame]
        Tuple of Pandas DataFrames with the normalized ratio and momentum indicator values.
    """

    ratio_data = calculate_relative_strength_ratio(symbols_data, benchmark_data)
    momentum_data = get_momentum(ratio_data, long_period, short_period)
    normalized_ratio = normalize(ratio_data, normalize_method)
    normalized_momentum = normalize(momentum_data, normalize_method)

    return normalized_ratio, normalized_momentum

def _create_figure_with_tails(
    ratios_data: DataFrame,
    momentum_data: DataFrame,
    study: str,
    benchmark_symbol: str,
    tail_periods: int,
    tail_interval: Literal["week", "month"]
):
    """Create the Plotly Figure Object with Tails."""
    color = 0
    symbols = ratios_data.columns.to_list()

    tail_dict = {"week": "W", "month": "M"}

    ratios_data.index = to_datetime(ratios_data.index)
    ratios_data =ratios_data.resample(tail_dict[tail_interval]).last()
    momentum_data.index = to_datetime(momentum_data.index)
    momentum_data = momentum_data.resample(tail_dict[tail_interval]).last()
    ratios_data = ratios_data.iloc[-tail_periods:]
    momentum_data = momentum_data.iloc[-tail_periods:]
    _tail_periods = len(ratios_data)
    tail_title = (
        f"The {_tail_periods} {tail_interval.capitalize()}s "
        f"Ending {ratios_data.index[-1].strftime('%Y-%m-%d')}"
    )
    x_min = ratios_data.min().min()
    x_max = ratios_data.max().max()
    y_min = momentum_data.min().min()
    y_max = momentum_data.max().max()
    # Create an empty list to store the scatter traces
    traces = []
    for symbol in symbols:

        # Select a single row from each dataframe
        x_data = ratios_data[symbol]
        y_data = momentum_data[symbol]
        name = symbol.upper().replace("^", "").replace(":US", "")
        # Create a trace for the line
        line_trace = go.Scattergl(
            x=x_data[:-1],  # All but the last data point
            y=y_data[:-1],  # All but the last data point
            mode="lines+markers",
            line=dict(color=color_sequence[color], width=2, dash="dash"),
            marker=dict(size=5, color=color_sequence[color]),
            opacity=0.3,
            showlegend=False,
            name=name,
            text=name,
            hovertemplate=
            "<b>%{fullData.name}</b>: " +
            "RS-Ratio: %{x:.4f}, " +
            "RS-Momentum: %{y:.4f}" +
            "<extra></extra>",
            hoverlabel=dict(font_size=10)
        )

        # Create a trace for the last data point
        marker_trace = go.Scatter(
            x=[x_data.iloc[-1]],  # Only the last data point
            y=[y_data.iloc[-1]],  # Only the last data point
            mode="markers+text",
            name=name,
            text=[name],
            textposition="middle center",
            textfont=dict(size=10, color="black") if len(name) < 4 else dict(size=8, color="black"),
            marker=dict(size=30, color=color_sequence[color], line=dict(color="black", width=1)),
            showlegend=False,
            hovertemplate=
            "<b>%{text}</b>: " +
            "RS-Ratio: %{x:.4f}, " +
            "RS-Momentum: %{y:.4f}" +
            "<extra></extra>",
        )
        traces.extend([line_trace, marker_trace])
        color += 1
    padding = 0.1
    y_range = [y_min - padding * abs(y_min) - 0.3, y_max + padding * abs(y_max) + 0.3]
    x_range = [x_min - padding * abs(x_min) - 0.3, x_max + padding * abs(x_max) + 0.3]

    layout = go.Layout(
        title={
            "text": (
                f"Relative Rotation Against {benchmark_symbol.replace('^', '')} {study.capitalize()} For {tail_title}"
            ),
            "x": 0.5,
            "xanchor": "center",
            "font": dict(color="white", size=18)
        },
        xaxis=dict(
            title="RS-Ratio",
            showgrid=True,
            zeroline=True,
            zerolinecolor="black",
            range=x_range,
            gridcolor="lightgrey",
            titlefont=dict(color="white", size=16),
            tickfont=dict(color="white"),
            showspikes=False,
        ),
        yaxis=dict(
            title="<br>RS-Momentum",
            showgrid=True,
            zeroline=True,
            zerolinecolor="black",
            range=y_range,
            gridcolor="lightgrey",
            titlefont=dict(color="white", size=16),
            tickfont=dict(color="white"),
            side="left",
            title_standoff=5,
        ),
        plot_bgcolor="rgba(255,255,255,1)",
        paper_bgcolor="rgba(0,0,0,5)",
        shapes=[
            go.layout.Shape(
                type="rect",
                xref="x",
                yref="y",
                x0=0,
                y0=0,
                x1=x_range[1],
                y1=y_range[1],
                fillcolor="lightgreen",
                opacity=0.3,
                layer="below",
                line_width=0,
            ),
            go.layout.Shape(
                type="rect",
                xref="x",
                yref="y",
                x0=x_range[0],
                y0=0,
                x1=0,
                y1=y_range[1],
                fillcolor="lightblue",
                opacity=0.3,
                layer="below",
                line_width=0,
            ),
            go.layout.Shape(
                type="rect",
                xref="x",
                yref="y",
                x0=x_range[0],
                y0=y_range[0],
                x1=0,
                y1=0,
                fillcolor="lightpink",
                opacity=0.3,
                layer="below",
                line_width=0,
            ),
            go.layout.Shape(
                type="rect",
                xref="x",
                yref="y",
                x0=0,
                y0=y_range[0],
                x1=x_range[1],
                y1=0,
                fillcolor="lightyellow",
                opacity=0.3,
                layer="below",
                line_width=0,
            ),
            go.layout.Shape(
                type="rect",
                xref="x",
                yref="y",
                x0=x_range[0],
                y0=y_range[0],
                x1=x_range[1],
                y1=y_range[1],
                line=dict(
                    color="Black",
                    width=1,
                ),
            fillcolor="rgba(0,0,0,0)",
            layer="above",
            ),
        ],
        annotations=[
            go.layout.Annotation(
                x=1,
                xref="paper",
                y=1,
                yref="paper",
                text="Leading",
                showarrow=False,
                font=dict(
                    size=18,
                    color="darkgreen",
                ),
            ),
            go.layout.Annotation(
                x=1,
                xref="paper",
                y=0,
                yref="paper",
                text="Weakening",
                showarrow=False,
                font=dict(
                    size=18,
                    color="goldenrod",
                ),
            ),
            go.layout.Annotation(
                x=0,
                xref="paper",
                y=0,
                yref="paper",
                text="Lagging",
                showarrow=False,
                font=dict(
                    size=18,
                    color="red",
                ),
            ),
            go.layout.Annotation(
                x=0,
                xref="paper",
                yref="paper",
                y=1,
                text="Improving",
                showarrow=False,
                font=dict(
                    size=18,
                    color="blue",
                ),
            ),
        ],
        autosize=True,
        margin=dict(
            l=30,
            r=50,
            b=50,
            t=50,
            pad=0,
        ),
        dragmode="pan",
        hovermode="closest",
        hoverlabel=dict(font=dict(color="white")),
    )

    fig = go.Figure(data=traces, layout=layout)

    return fig


def _create_figure(
    ratios_data: DataFrame,
    momentum_data: DataFrame,
    benchmark_symbol: str,
    date: Optional[dateType] = None,
    study: Literal["price", "volume", "volatility"] = "price",
):
    """Create the Plotly Figure Object without Tails."""

    if date is not None:
        date = date.strftime("%Y-%m-%d") if isinstance(date, dateType) else date  # type: ignore
        if date not in ratios_data.index:
            _warn(f"Date {date} not found in data, using the last available date.")
            date = ratios_data.index[-1].strftime("%Y-%m-%d") if isinstance(date, dateType) else ratios_data.index[-1]
    if date is None:
        date = ratios_data.index[-1].strftime("%Y-%m-%d") if isinstance(date, dateType) else ratios_data.index[-1]

    # Select a single row from each dataframe
    row_x = ratios_data.loc[date]
    row_y = momentum_data.loc[date]

    x_max = row_x.max() + 0.5
    x_min = row_x.min() - 0.5
    y_max = row_y.max() + 0.5
    y_min = row_y.min() - 0.5

    # Create an empty list to store the scatter traces
    traces = []

    # Loop through each column in the row_x dataframe
    for i, (column_name, value_x) in enumerate(row_x.items()):
        # Retrieve the corresponding value from the row_y dataframe
        value_y = row_y[column_name]
        marker_name = column_name.upper().replace("^", "").replace(":US", "")
        # Create a scatter trace for each column
        trace = go.Scatter(
            x=[value_x],
            y=[value_y],
            mode="markers+text",
            text=[marker_name],
            textposition="middle center",
            textfont=dict(size=10 if len(marker_name) < 4 else 8, color="black"),
            marker=dict(size= 30, color=color_sequence[i % len(color_sequence)], line=dict(color="black", width=1)),
            name=column_name,
            showlegend=False,
            hovertemplate=
            "<b>%{text}</b>: " +
            "RS-Ratio: %{x:.4f}, " +
            "RS-Momentum: %{y:.4f}" +
            "<extra></extra>",
        )
        # Add the trace to the list
        traces.append(trace)

    padding = 0.1
    y_range = [y_min - padding * abs(y_min) - 0.3, y_max + padding * abs(y_max)]
    x_range = [x_min - padding * abs(x_min), x_max + padding * abs(x_max)]


    layout = go.Layout(
        title={
            "text": (
                f"RS-Ratio vs RS-Momentum of {study.capitalize()} "
                f"Against {benchmark_symbol.replace('^', '')} - {to_datetime(row_x.name).strftime('%Y-%m-%d')}"
            ),
            "x": 0.5,
            "xanchor": "center",
            "font": dict(color="white", size=20)
        },
        xaxis=dict(
            title="RS-Ratio",
            showgrid=True,
            zeroline=True,
            zerolinecolor="black",
            range=x_range,
            gridcolor="lightgrey",
            titlefont=dict(color="white", size=16),
            tickfont=dict(color="white"),
            showspikes=False,
        ),
        yaxis=dict(
            title="<br>RS-Momentum",
            showgrid=True,
            zeroline=True,
            zerolinecolor="black",
            range=y_range,
            gridcolor="lightgrey",
            titlefont=dict(color="white", size=16),
            tickfont=dict(color="white"),
            side="left",
            title_standoff=5,
        ),
        plot_bgcolor="rgba(255,255,255,1)",
        paper_bgcolor="rgba(0,0,0,5)",
        shapes=[
            go.layout.Shape(
                type="rect",
                xref="x",
                yref="y",
                x0=0,
                y0=0,
                x1=x_range[1],
                y1=y_range[1],
                fillcolor="lightgreen",
                opacity=0.3,
                layer="below",
                line_width=0,
            ),
            go.layout.Shape(
                type="rect",
                xref="x",
                yref="y",
                x0=x_range[0],
                y0=0,
                x1=0,
                y1=y_range[1],
                fillcolor="lightblue",
                opacity=0.3,
                layer="below",
                line_width=0,
            ),
            go.layout.Shape(
                type="rect",
                xref="x",
                yref="y",
                x0=x_range[0],
                y0=y_range[0],
                x1=0,
                y1=0,
                fillcolor="lightpink",
                opacity=0.3,
                layer="below",
                line_width=0,
            ),
            go.layout.Shape(
                type="rect",
                xref="x",
                yref="y",
                x0=0,
                y0=y_range[0],
                x1=x_range[1],
                y1=0,
                fillcolor="lightyellow",
                opacity=0.3,
                layer="below",
                line_width=0,
            ),
            go.layout.Shape(
                type="rect",
                xref="x",
                yref="y",
                x0=x_range[0],
                y0=y_range[0],
                x1=x_range[1],
                y1=y_range[1],
                line=dict(
                    color="Black",
                    width=1,
                ),
            fillcolor="rgba(0,0,0,0)",
            layer="above",
            ),
        ],
        annotations=[
            go.layout.Annotation(
                x=1,
                xref="paper",
                y=1,
                yref="paper",
                text="Leading",
                showarrow=False,
                font=dict(
                    size=18,
                    color="darkgreen",
                ),
            ),
            go.layout.Annotation(
                x=1,
                xref="paper",
                y=0,
                yref="paper",
                text="Weakening",
                showarrow=False,
                font=dict(
                    size=18,
                    color="goldenrod",
                ),
            ),
            go.layout.Annotation(
                x=0,
                xref="paper",
                y=0,
                yref="paper",
                text="Lagging",
                showarrow=False,
                font=dict(
                    size=18,
                    color="red",
                ),
            ),
            go.layout.Annotation(
                x=0,
                xref="paper",
                yref="paper",
                y=1,
                text="Improving",
                showarrow=False,
                font=dict(
                    size=18,
                    color="blue",
                ),
            ),
        ],
        autosize=True,
        margin=dict(
            l=30,
            r=50,
            b=50,
            t=50,
            pad=0,
        ),
        dragmode="pan",
        hovermode="x unified",
        hoverlabel=dict(font=dict(color="white")),
    )

    fig = go.Figure(data=traces, layout=layout)

    return fig


class RelativeRotationData(Data):
    """Relative Roration Data Model."""

    symbols: List[str]
    benchmark: str
    study: Literal["price", "volume", "volatility"] = "price"
    date: Optional[dateType] = None
    long_period: Optional[int] = 252
    short_period: Optional[int] = 21
    window: Optional[int] = 21
    trading_periods: Optional[int] = 252
    normalize_method: Optional[Literal["z", "m", "a"]] = "z"
    tail_periods: Optional[int] = 30
    tail_interval: Literal["week", "month"] = "week"
    provider: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    symbols_data: Optional[List[Data]] = None
    benchmark_data: Optional[List[Data]] = None
    rs_ratios: Optional[List[Data]] = None
    rs_momentum: Optional[List[Data]] = None


    def show(
        self,
        date: Optional[dateType] = None,
        show_tails: bool = True,
        tail_periods: Optional[int] = 12,
        tail_interval: Literal["week", "month", None] = "week",
        external: bool = False,
        **kwargs,
    ) -> Union[OpenBBFigure, None]:
        """Plot the data, optionally for a specific date in the collected data."""
        ratios_data = basemodel_to_df(self.rs_ratios).set_index("date")  # type: ignore
        ratios_data.index = to_datetime(ratios_data.index)
        momentum_data = basemodel_to_df(self.rs_momentum).set_index("date")  # type: ignore
        momentum_data.index = to_datetime(momentum_data.index)
        if date is not None:
            show_tails = False
        if show_tails is False:
            fig = _create_figure(
                ratios_data = ratios_data,
                momentum_data = momentum_data,
                benchmark_symbol =self.benchmark,
                date=date if date is not None else self.date,
                study=self.study,
            )
        if show_tails is True:
            fig = _create_figure_with_tails(
                ratios_data = ratios_data,
                momentum_data = momentum_data,
                benchmark_symbol =self.benchmark,
                study=self.study,
                tail_periods= self.tail_periods if tail_periods is None else tail_periods,  # type: ignore
                tail_interval= self.tail_interval if tail_interval is None else tail_interval,
            )
        return (
            OpenBBFigure(fig, create_backend=True).show() if external is False  # type: ignore
            else OpenBBFigure(fig, create_backend=True)  # type: ignore
        )


async def create(
    symbols: Union[List[str], DataFrame, Data],
    benchmark: Union[str, DataFrame, Data],
    study: Optional[Literal["price", "volume", "volatility"]] = "price",
    date: Optional[dateType] = None,
    long_period: Optional[int] = 252,
    short_period: Optional[int] = 21,
    window: Optional[int] = 21,
    trading_periods: Optional[int] = 252,
    normalize_method: Optional[Literal["z", "m", "a"]] = "z",
    tail_periods: Optional[int] = 30,
    tail_interval: Literal["week", "month"] = "week",
    provider:Optional[str] = None,
):
    """
    Create an instance of RelativeRotationData with provided data or a list of symbols and the benchmark.
    """

    try:
        symbols = basemodel_to_df(symbols).set_index("date")  # type: ignore
        benchmark = basemodel_to_df(benchmark).set_index("date")  # type: ignore
    except AttributeError:
        pass

    if (
        isinstance(symbols, list)
        and all(isinstance(s, str) for s in symbols)
        and isinstance(benchmark, str)
    ):

        self = RelativeRotationData(
            symbols=symbols,
            benchmark=benchmark,
            study=study,  # type: ignore
            date=date,
            long_period=long_period,
            short_period=short_period,
            window=window,
            trading_periods=trading_periods,
            normalize_method=normalize_method,
            tail_periods=tail_periods,
            tail_interval=tail_interval,
            provider=provider
        )
        await _fetch_data(self)

    if (
        isinstance(symbols, Data)
        and isinstance(benchmark, Data)
    ):
        symbols = basemodel_to_df(symbols).set_index("date")
        benchmark = basemodel_to_df(benchmark).set_index("date")

    if (
        isinstance(symbols, List)
        and all(isinstance(s, dict) for s in symbols)
        and isinstance(benchmark, List)
        and all(isinstance(s, dict) for s in benchmark)
    ):
        try:
            symbols = DataFrame(symbols).set_index("date")
            benchmark = DataFrame(benchmark).set_index("date")
        except ValueError:
            pass


    if isinstance(benchmark, Series):
        benchmark = benchmark.to_frame()

    if isinstance(symbols, DataFrame) and isinstance(benchmark, DataFrame):
        if "date" in symbols.columns:
            symbols = symbols.set_index("date")
        if "date" in benchmark.columns:
            benchmark = benchmark.set_index("date")
        self = RelativeRotationData(
            symbols=symbols.columns.to_list(),
            benchmark=benchmark.columns.to_list()[0],
            study=study,  # type: ignore
            date=date,
            long_period=long_period,
            short_period=short_period,
            window=window,
            trading_periods=trading_periods,
            normalize_method=normalize_method,
            tail_periods=tail_periods,
            tail_interval=tail_interval,
        )
        self.symbols_data = symbols  # type: ignore
        self.benchmark_data = benchmark  # type: ignore
    if len(self.symbols_data) <= 252 and self.study in ["price", "volume"]:  # type: ignore
        raise ValueError(
            "Supplied data must have more than one year of back data to calculate"
            " the most recent day in the time series."
        )
    if self.study == "volatility" and len(self.symbols_data) <= 504:  # type: ignore
        raise ValueError(
            "Supplied data must have more than two years of back data to calculate"
            " the most recent day in the time series as a volatility study."
        )
    if len(self.symbols_data.index) != len(self.benchmark_data.index):  # type: ignore
        raise ValueError("Supplied data must have the same index.")

    await _process_data(self)  # type: ignore
    self.symbols_data = df_to_basemodel(self.symbols_data.reset_index())  # type: ignore
    self.benchmark_data = df_to_basemodel(self.benchmark_data.reset_index())  # type: ignore

    return self # type: ignore

async def _fetch_data(self):
    """Fetch the data."""
    if self.provider is None:
        _warn("Provider was not specified. Using default provider: yfinance.")
        self.provider = "yfinance"
    df1, df2 = await get_data(
        symbols = self.symbols,
        benchmark = self.benchmark,
        study = self.study,
        date = self.date,
        provider=self.provider,
        tail_periods=self.tail_periods,
        tail_interval=self.tail_interval
    )
    self.symbols_data =  df1
    self.benchmark_data = df2
    return self

async def _process_data(self):
    """Process the data."""
    if self.study == "volatility":
        self.symbols_data = standard_deviation(
            self.symbols_data,
            window = self.window,
            trading_periods = self.trading_periods,
        )
        self.benchmark_data = standard_deviation(
            self.benchmark_data,
            window = self.window,
            trading_periods = self.trading_periods,
        )
    ratios, momentum = process_data(
        self.symbols_data,
        self.benchmark_data,
        long_period=self.long_period,
        short_period=self.short_period,
        normalize_method=self.normalize_method,
    )
    # Re-index rs_ratios using the new index
    index_after_dropping_nans = momentum.dropna().index
    ratios = ratios.reindex(index_after_dropping_nans)
    self.rs_ratios = df_to_basemodel(ratios.reset_index())
    self.rs_momentum = df_to_basemodel(momentum.dropna().reset_index())
    self.end_date = to_datetime(ratios.index[-1]).strftime("%Y-%m-%d")
    self.start_date = to_datetime(ratios.index[0]).strftime("%Y-%m-%d")
    return self


SPDRS = [
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XRT",
    "XHB",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
    "XLRE",
]

class RelativeRotation(OBBject):
    """
    Relative Rotation OBBject.

    Calculated results and the source data are stored in the `results` attribute.

    Results
    -------
    symbols: List[str]
        The list of symbols suppied.
    benchmark: str
        The benchmark symbol supplied.
    study: str
        The type of data selected. Default is "price".
    date: dateType
        The target end date of the study. If None, the current date is used.
    long_period: int
        The length of the long period for momentum calculation, as entered.
        Default is 252.
    short_period: int
        The length of the short period for momentum calculation, as entered.
        Default is 21.
    window: int
        The length of window for the standard deviation calculation, as entered.
        Default is 21. Only valid of study is "volatility".
    trading_periods: int
        The number of trading periods per year, as entered. Default is 252.
        Only valid of study is "volatility".
    normalize_method: str
        The normalization method selected. Default is "z".
            Z: Z-Score Standardization
            M: Min/Max Scaling
            A: Absolute Maximum Scale
    tail_periods: int
        The number of periods to show tails for each asset, as entered.
        Supplied values determine the overall length of the data.
        This value determines the maximum `tail_periods` in the `show()` method.
        Parameter is relative to `tail_interval`.
        Default is 30 weeks for fetching and 12 for display.
    tail_interval: str
        The interval to show tails for each asset, as entered. Default is "week".
        None is a proxy for "week".
    provider: str
        The data provider when fetching data.
    start_date: str
        The start date of the data, after processing.
    end_date: str
        The end date of the data, after processing.
    symbols_data: List[Data]
        The raw data for each symbol, relative to the 'study' parameter.
    benchmark_data: List[Data]
    rs_ratios: List[Data]
        The normalized relative strength ratios data.
    rs_momentum: List[Data]
        The normalized relative strength momentum data.
    """

    def show(
        self,
        date: Optional[dateType] = None,
        show_tails: bool = True,
        tail_periods: Optional[int] = None,
        tail_interval: Literal["week", "month", None] = None,
        external: bool = False,
        **kwargs,
    ) -> Union[OpenBBFigure, None]:
        """
        Create and display the Relative Rotation Graph.

        This modifies the `chart` attribute of the orignal object.

        Parameters
        ----------
        date: Optional[datetime.date] = None
            The date to show the Relative Rotation Graph. If None, the most recent date in the
            data is used. Specifying a date will override 'tail' parameters. Defaults to None.
            Format as YYYY-MM-DD.
        show_tails: bool = True
            Whether to show the tails of the Relative Rotation Graph. Defaults to True.
            Specifying a date is akin to False.
        tail_periods: Optional[int] = 12
            The number of periods to show tails for each asset. The number is relative
            to the choice for `tail_interval`. Defaults to 12. The maximimum is limited
            to the original data length, by default will be 30 weeks. For longer data,
            rerun the function with the `tail_periods` set to a higher number.
        tail_interval: Literal["week", "month", None] = "week"
            The interval to show tails for. Defaults to "week". None is a proxy for "week".
        external: bool = False
            When True, the OpenBBFigure Object is returned instead of being displayed.

        Returns
        -------
        Union[OpenBBFigure, None]
            If `external` is True, returns the OpenBBFigure Object. Otherwise, returns None.
            Figures are displayed in a PyWry window from the command line.
        """
        if date is not None:
            show_tails = False
        if tail_periods is None:
            tail_periods = self.results.tail_periods  # type: ignore
        if tail_interval is None:
            tail_interval = self.results.tail_interval  # type: ignore

        fig = self.results.show(  # type: ignore
            date=date,
            show_tails=show_tails,
            tail_periods=tail_periods,
            tail_interval=tail_interval,
            external=True,
        )
        content = fig.to_plotly_json()
        format = ChartFormat.plotly
        chart = Chart(content=content, format=format, fig=fig)
        self.chart = chart
        self.provider = self.results.provider  # type: ignore

        return fig.show() if external is False else fig

    def to_chart(
        self,
        date: Optional[dateType] = None,
        show_tails: bool = True,
        tail_periods: Optional[int] = None,
        tail_interval: Literal["week", "month", None] = None,
        **kwargs,
    ):
        """
        Creates the Relative Rotation Graph returning to the `chart` attribute of the orignal object.

        Proxy function for `show()` that does not display. Use `show()` to display the chart.

        This modifies the `chart` attribute of the orignal object.

        Parameters
        ----------
        date: Optional[datetime.date] = None
            The date to show the Relative Rotation Graph. If None, the most recent date in the
            data is used. Specifying a date will override 'tail' parameters. Defaults to None.
            Format as YYYY-MM-DD.
        show_tails: bool = True
            Whether to show the tails of the Relative Rotation Graph. Defaults to True.
            Specifying a date is akin to False.
        tail_periods: Optional[int] = 12
            The number of periods to show tails for each asset. The number is relative
            to the choice for `tail_interval`. Defaults to 12. The maximimum is limited
            to the original data length, by default will be 30 weeks. For longer data,
            rerun the function with the `tail_periods` set to a higher number.
        tail_interval: Literal["week", "month", None] = "week"
            The interval to show tails for. Defaults to "week". If None, it will default to
            12 weeks.

        Returns
        -------
        Self
            The Relative Rotation Graph is returned to the `chart` attribute of the orignal object.
        """
        if date is not None:
            show_tails = False
        if tail_periods is None:
            tail_periods = self.results.tail_periods  # type: ignore
        if tail_interval is None:
            tail_interval = self.results.tail_interval  # type: ignore

        fig = self.results.show(  # type: ignore
            date=date,
            show_tails=show_tails,
            tail_periods=tail_periods,
            tail_interval=tail_interval,
            external=True,
        )
        content = fig.to_plotly_json()
        format = ChartFormat.plotly
        chart = Chart(content=content, format=format, fig=fig)
        self.chart = chart
        self.provider = self.results.provider  # type: ignore

        return self
