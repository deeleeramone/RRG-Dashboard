"""Relative Rotation Studies - powered by openbb-technical."""

from datetime import date as dateType, datetime, timedelta
from typing import List, Literal, Optional, Union

from openbb_charting.charts.relative_rotation import (
    create_rrg_with_tails,
    create_rrg_without_tails,
)
from openbb_charting.core.openbb_figure import OpenBBFigure
from openbb_core.app.command_runner import CommandRunner
from openbb_core.app.model.charts.chart import Chart
from openbb_core.app.model.obbject import OBBject
from openbb_core.app.utils import basemodel_to_df, df_to_basemodel
from openbb_core.provider.abstract.data import Data
from openbb_technical.relative_rotation import (
    RelativeRotation as _RelativeRotation,
    RelativeRotationData as _RelativeRotationData,
)
from pandas import DataFrame, to_datetime


async def get_data(
    symbols: List[str],
    benchmark: str,
    date: Optional[dateType] = None,
    tail_periods: Optional[int] = 16,
    tail_interval: Literal["week", "month"] = "week",
) -> OBBject:
    """
    Fetch data with the OpenBB Platform Command Runner using yFinance.

    Parameters
    ----------
    symbols: List[str]
        List of symbols to fetch data for.
    benchmark: str
        Benchmark ticker symbol to compare against.
    study: Literal["price", "volume", "volatility"]
        The type of data to study. Default is "price".
    date: Optional[dateType]
        The date representing the end date of the study. Default is today.
    tail_periods: Optional[int]
        Number of tail periods for display. Used to calculate backfill.
    tail_interval: Literal["week", "month"]
        Interval for tail display. Used to calculate backfill.
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
    else:
        start_date = (datetime.now() - timedelta(weeks=backfill)).date()
        end_date = datetime.now().date()

    tickers = symbols + [benchmark]
    runner = CommandRunner()
    try:
        data = await runner.run(
            "/equity/price/historical",
            provider_choices={"provider": "yfinance"},
            standard_params={
                "symbol": ",".join(tickers),
                "start_date": start_date,
                "end_date": end_date,
                "interval": "1d",
            },
            extra_params={},
        )
    except Exception as e:
        raise RuntimeError(f"There was an error loading data for {symbols}: {e}")

    return data


SPDRS = [
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XHB",
    "XLU",
    "XLV",
    "XLY",
    "XLRE",
]


class RelativeRotation(OBBject):
    """Relative Rotation OBBject."""

    _tail_periods: int = 12
    _tail_interval: str = "week"

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

        Parameters
        ----------
        date: Optional[datetime.date]
            The date to show the graph for. Overrides tail parameters. Default is None.
        show_tails: bool
            Whether to show tails. Default is True.
        tail_periods: Optional[int]
            Number of periods for tails. Default uses the value from creation.
        tail_interval: Literal["week", "month", None]
            Interval for tails. Default uses the value from creation.
        external: bool
            When True, returns the OpenBBFigure instead of displaying.

        Returns
        -------
        Union[OpenBBFigure, None]
        """
        results = self.results
        ratios_data = basemodel_to_df(results.rs_ratios).set_index("date")
        ratios_data.index = to_datetime(ratios_data.index)
        momentum_data = basemodel_to_df(results.rs_momentum).set_index("date")
        momentum_data.index = to_datetime(momentum_data.index)

        if tail_periods is None:
            tail_periods = getattr(self, "_tail_periods", 12)
        if tail_interval is None:
            tail_interval = getattr(self, "_tail_interval", "week")
        if date is not None:
            show_tails = False

        if show_tails:
            fig = create_rrg_with_tails(
                ratios_data=ratios_data,
                momentum_data=momentum_data,
                benchmark_symbol=results.benchmark,
                study=results.study,
                tail_periods=tail_periods,
                tail_interval=tail_interval,
            )
        else:
            fig = create_rrg_without_tails(
                ratios_data=ratios_data,
                momentum_data=momentum_data,
                benchmark_symbol=results.benchmark,
                study=results.study,
                date=date,
            )

        content = fig.to_plotly_json()
        chart = Chart(content=content, format="plotly", fig=fig)
        self.chart = chart
        self.provider = "yfinance"

        if external:
            return OpenBBFigure(fig, create_backend=True)
        return OpenBBFigure(fig, create_backend=True).show()

    def to_chart(
        self,
        date: Optional[dateType] = None,
        show_tails: bool = True,
        tail_periods: Optional[int] = None,
        tail_interval: Literal["week", "month", None] = None,
        **kwargs,
    ):
        """Create the chart without displaying. Use show() to display."""
        self.show(
            date=date,
            show_tails=show_tails,
            tail_periods=tail_periods,
            tail_interval=tail_interval,
            external=True,
        )
        return self


async def create(
    symbols: Union[List[str], DataFrame, Data],
    benchmark: Union[str, DataFrame, Data],
    study: Optional[Literal["price", "volume", "volatility"]] = "price",
    date: Optional[dateType] = None,
    long_period: Optional[int] = 252,
    short_period: Optional[int] = 21,
    window: Optional[int] = 21,
    trading_periods: Optional[int] = 252,
    tail_periods: Optional[int] = 30,
    tail_interval: Literal["week", "month"] = "week",
):
    """
    Create a RelativeRotation instance.

    Fetches data via yFinance if symbols/benchmark are strings,
    or processes supplied DataFrame/Data directly.
    """
    raw_data = None

    if (
        isinstance(symbols, list)
        and all(isinstance(s, str) for s in symbols)
        and isinstance(benchmark, str)
    ):
        raw_data = await get_data(
            symbols=symbols,
            benchmark=benchmark,
            date=date,
            tail_periods=tail_periods,
            tail_interval=tail_interval,
        )
        data_for_rr = raw_data.results
    else:
        # Handle pre-supplied data
        if isinstance(symbols, Data) and isinstance(benchmark, Data):
            symbols = basemodel_to_df(symbols).set_index("date")
            benchmark = basemodel_to_df(benchmark).set_index("date")

        if isinstance(symbols, DataFrame) and isinstance(benchmark, DataFrame):
            if "date" in symbols.columns:
                symbols = symbols.set_index("date")
            if "date" in benchmark.columns:
                benchmark = benchmark.set_index("date")
            combined = symbols.join(benchmark)
            data_for_rr = df_to_basemodel(combined.reset_index())
            benchmark = benchmark.columns[0]
        else:
            raise ValueError("Unsupported data format for symbols/benchmark.")

    # Use openbb-technical's RelativeRotation for all calculations
    rr = _RelativeRotation(
        data=data_for_rr,
        benchmark=benchmark if isinstance(benchmark, str) else benchmark,
        study=study,
        long_period=long_period,
        short_period=short_period,
        window=window,
        trading_periods=trading_periods,
    )

    # Build the result data model
    result = _RelativeRotationData.model_validate(rr.__dict__)

    # Wrap in our OBBject subclass
    obj = RelativeRotation(results=result)
    obj._tail_periods = tail_periods or 12
    obj._tail_interval = tail_interval

    return obj
