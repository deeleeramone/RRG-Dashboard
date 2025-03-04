import asyncio
import importlib.util
from datetime import datetime

import pandas as pd
from openbb_core.app.utils import basemodel_to_df

import streamlit as st

st.set_page_config(
    layout="wide",
    page_title="Relative Rotation",
    initial_sidebar_state="expanded",
)


symbols = []
benchmark = ""
STUDY_CHOICES = ["Price", "Volume", "Volatility"]
SOURCE_CHOICES = ["Yahoo Finance", "Cboe"]
source_input = "Yahoo Finance"
source_dict = {"Yahoo Finance": "yfinance", "Cboe": "cboe"}
source = source_dict[source_input]
window_input = 21
study = "price"
short_period = 21
long_period = 252
window = 21
trading_periods_input = 252
trading_periods = 252
tail_interval = "week"
tail_interval_input = "week"
tail_periods_input = 30
tail_periods = 30
show_tails = False
st.session_state.rrg_data = st.empty()
st.session_state.date = None
st.session_state.fig = None
rrg_data = st.empty()
st.session_state.data_tables = None


def import_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


module = import_from_file("module", "relative_rotation.py")

st.sidebar.markdown(
    """
<style>
section[data-testid="stSidebar"] {
    top: 1% !important;
    height: 98.25% !important;
    left: 0.33% !important;
    margin-top: 0 !important;
}
section[data-testid="stSidebar"] img {
    margin-top: -75px !important;
    margin-left: -10px !important;
    width: 95% !important;
}
section[data-testid="stVerticalBlock"] {
    gap: 0rem;
}
body {
    line-height: 1.2;
}
</style>
<figure style='text-align: center;'>
    <img src='https://openbb.co/assets/images/ogimages/Homepage.png' />
    <figcaption style='font-size: 0.8em; color: #888;'>Powered by Open Source</figcaption>
</figure>
""",
    unsafe_allow_html=True,
)

st.sidebar.header("Relative Rotation Graph")

with st.sidebar:
    source_input = st.selectbox("Data Source", SOURCE_CHOICES, index=0, key="source")
r2c1, r2c2 = st.sidebar.columns([1, 1])

with r2c1:
    input_string = st.text_input(
        "Symbols", value=",".join(module.SPDRS), key="tickers"
    ).replace(" ", "")
    if input_string == "":
        st.write("Enter a list of tickers")

with r2c2:
    benchmark_input = st.text_input("Benchmark", value="SPY", key="benchmark")
    if benchmark_input == "":
        st.write("Enter a benchmark")

date_input = st.sidebar.date_input(
    "Target End Date", value=datetime.today(), key="date_input"
)
st.session_state.date = date_input

with st.sidebar:
    study_input = st.sidebar.selectbox("Study", STUDY_CHOICES, key="study")
    if study_input == "Volatility":
        st.sidebar.header("Volatility Annualization")
        r4c1, r4c2 = st.sidebar.columns([1, 1])
        with r4c1:
            window_input = st.number_input(
                "Rolling Window", min_value=0, value=21, key="window"
            )
        with r4c2:
            trading_periods_input = st.number_input(
                "Periods Per Year", min_value=0, value=252, key="trading_periods"
            )

st.sidebar.header("Long/Short Momentum Periods")

r3c1, r3c2 = st.sidebar.columns([1, 1])  # Create a new set of columns

with r3c1:
    long_period_input = st.number_input(
        "Long Period", min_value=0, value=252, key="long_period"
    )

with r3c2:
    short_period_input = st.number_input(
        "Short Period", min_value=0, value=21, key="short_period"
    )

# Initialize the session state for the button if it doesn't exist
if "button_clicked" not in st.session_state:
    st.session_state.button_clicked = False

# When the button is clicked, update the session state
if st.sidebar.button("Fetch Data"):
    st.session_state.button_clicked = True

with st.sidebar:
    show_tails_input = st.checkbox("Show Tails", value=False, key="show_tails")
    r5c1, r5c2 = st.sidebar.columns([1, 1])
    with r5c1:
        tail_periods_input = st.number_input(
            "Tail Periods", min_value=0, value=30, key="tail_periods"
        )
    with r5c2:
        tail_interval_input = st.selectbox(
            "Tail Interval", ["Week", "Month"], key="tail_interval", index=0
        )


symbols = input_string.upper().split(",")
benchmark = benchmark_input.split(",")[0].upper()
study = study_input.lower()
long_period = long_period_input
short_period = short_period_input
show_tails = show_tails_input
tail_periods = tail_periods_input
tail_interval = tail_interval_input.lower()
window = window_input
trading_periods = trading_periods_input
date = date_input
source = source_dict[source_input]

if st.session_state.button_clicked:
    try:
        rrg_data = asyncio.run(
            module.create(
                symbols=symbols,
                benchmark=benchmark,
                study=study,
                date=pd.to_datetime(date),
                long_period=long_period,
                short_period=short_period,
                window=window,
                trading_periods=trading_periods,
                tail_periods=tail_periods,
                tail_interval=tail_interval,
                provider=source,
            )
        )
        st.session_state.rrg_data = rrg_data
        st.session_state.first_run = False
    except Exception:
        st.session_state.rrg_data = None
        st.session_state.first_run = True
        if input_string != "" and benchmark_input != "":
            st.write(
                "There was an error fetching the data."
                " Please check if the symbols are correct and available at the source."
                " Volume data may not exist for most indexes, for example."
            )
            st.write(str(Exception.args[0]))
        if input_string == "" or benchmark_input == "":
            st.write("Please enter a list of symbols and a benchmark.")


main_chart = st.expander("Relative Rotation Graph", expanded=True)

if "first_run" not in st.session_state:
    st.session_state.first_run = True

if not st.session_state.first_run and st.session_state.rrg_data is not None:
    with main_chart:
        fig = (
            st.session_state.rrg_data.show(
                date, show_tails, tail_periods, tail_interval, external=True
            )
            if show_tails is False
            else st.session_state.rrg_data.show(
                show_tails=show_tails,
                tail_periods=tail_periods,
                tail_interval=tail_interval,
                external=True,
            )
        )
        fig.update_layout(height=600, margin=dict(l=0, r=20, b=0, t=50, pad=0))
        st.session_state.fig = fig
        st.plotly_chart(
            st.session_state.fig,
            config={
                "scrollZoom": True,
                "displaylogo": False,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                "modeBarButtons": [
                    ["toImage"],
                    ["zoomIn2d", "zoomOut2d", "autoScale2d", "zoom2d", "pan2d"],
                ],
            },
        )
        st.markdown(
            """
            <style>
            .js-plotly-plot .plotly .modebar {
                top: -40px !important;
                right: 30px !important;
                bottom: auto !important;
                transform: translateY(0) !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("Study Data Table", expanded=False):
        symbols_data = (
            basemodel_to_df(st.session_state.rrg_data.symbols_data).join(
                basemodel_to_df(st.session_state.rrg_data.benchmark_data)[
                    st.session_state.rrg_data.benchmark
                ]
            )
        ).set_index("date")
        symbols_data.index = pd.to_datetime(symbols_data.index).strftime("%Y-%m-%d")
        st.dataframe(symbols_data)

    with st.expander("Relative Strength Ratio Table", expanded=False):
        ratios_data = basemodel_to_df(st.session_state.rrg_data.rs_ratios).set_index(
            "date"
        )
        ratios_data.index = pd.to_datetime(ratios_data.index).strftime("%Y-%m-%d")
        st.dataframe(ratios_data)

    with st.expander("Relative Strength Momentum Table", expanded=False):
        ratios_data = basemodel_to_df(st.session_state.rrg_data.rs_momentum).set_index(
            "date"
        )
        ratios_data.index = pd.to_datetime(ratios_data.index).strftime("%Y-%m-%d")
        st.dataframe(ratios_data)

with st.expander("About"):
    st.write(
        """
        This dashboard is powered by the OpenBB Platform. More information can be found here: https://docs.openbb.co/platform

        A Relative Rotation Graph is a study of the Relative Strength Ratio vs. Relative Strength Momentum against a
        benchmark. They are lagging indicators and are typically used for comparing sector or index
        constituents.

        The Relative Strength Ratio is the price (volume or realized volatility) of the asset divided by the
        benchmark.

        The Relative Strength (RS) Momentum is the momentum of the Relative Strength Ratio.
        In this application, momentum is calculated as the trailing 12-month minus 1-month return.
        The default values for long and short periods are 252 and 21, which is the number of trading
        days in a year and a month, respectively. These values can be changed in the sidebar.

        Realized volatility is calculated as the annualized standard deviation over a trailing 1-month period.
        The default values of 252 and 21 can be changed in the sidebar.

        All calculations are daily closing values from the source selected in the sidebar. It should not be assumed
        that volume represents 100% market coverage. This dashboard is for demonstration purposes only and
        should not be used to make inferences or investment decisions.
    """
    )
