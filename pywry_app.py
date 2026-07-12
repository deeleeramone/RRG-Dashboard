"""Relative Rotation Graph Dashboard - PyWry Native App."""

import asyncio
import importlib.util
import time as _time
from datetime import datetime

import pandas as pd
from openbb_core.app.utils import basemodel_to_df
from pywry import (
    Button,
    Checkbox,
    DateInput,
    Div,
    HtmlContent,
    NumberInput,
    Option,
    PyWry,
    Select,
    TextInput,
    Toggle,
    Toolbar,
    WindowMode,
)


def import_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


module = import_from_file("module", "relative_rotation.py")

SIDEBAR_CSS = """
.pywry-toolbar-left {
    width: 300px !important;
    min-width: 300px !important;
    max-width: 300px !important;
}
.pywry-toolbar-left .pywry-toolbar-content {
    padding: 16px 14px !important;
    gap: 6px !important;
    width: 100% !important;
}
.pywry-toolbar-left .pywry-toolbar-content > * {
    width: 100% !important;
    max-width: 100% !important;
    flex-shrink: 0 !important;
}
.pywry-toolbar-left .pywry-div {
    flex-wrap: nowrap !important;
    flex-direction: column !important;
    width: 100% !important;
    padding: 0 !important;
}
.pywry-toolbar-left .pywry-div[style*="flex-direction:row"],
.pywry-toolbar-left .pywry-div[style*="flex-direction: row"] {
    flex-direction: row !important;
    flex-wrap: nowrap !important;
}
.pywry-toolbar-left .pywry-div[style*="flex-direction:row"] > *,
.pywry-toolbar-left .pywry-div[style*="flex-direction: row"] > * {
    flex: 1 1 0% !important;
    min-width: 0 !important;
}
.pywry-toolbar-left .pywry-input-group {
    width: 100% !important;
    margin-bottom: 2px !important;
    flex-direction: column !important;
}
.pywry-toolbar-left .pywry-input-group label {
    font-size: 11px !important;
    color: #999 !important;
    margin-bottom: 2px !important;
}
.pywry-toolbar-left .pywry-input,
.pywry-toolbar-left .pywry-dropdown,
.pywry-toolbar-left select,
.pywry-toolbar-left input[type="text"],
.pywry-toolbar-left input[type="number"],
.pywry-toolbar-left input[type="date"] {
    width: 100% !important;
    box-sizing: border-box !important;
    font-size: 13px !important;
}
.pywry-toolbar-left .pywry-btn {
    width: 100% !important;
    box-sizing: border-box !important;
}
.sidebar-heading {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: var(--pywry-text-primary, #e0e0e0) !important;
    padding: 10px 0 4px 0 !important;
    margin: 0 !important;
    width: 100% !important;
}
.sidebar-rule {
    border: none !important;
    border-top: 1px solid var(--pywry-border-color, rgba(255,255,255,0.08)) !important;
    margin: 6px 0 2px 0 !important;
    width: 100% !important;
}
/* Hide plotly modebar completely */
.modebar-container { display: none !important; }
.modebar { display: none !important; }
"""


# ── State ─────────────────────────────────────────────────────────────
rrg_data = None
_show_tails = False
_dark_mode = True
_current_study = "price"
_chart_shown = False  # True after first show_plotly call
_FETCH_KEYS = {"symbols", "benchmark", "study", "long_period",
               "short_period", "window", "trading_periods"}
_last_fetch_params = {}
_last_rebuild = 0.0

app = PyWry(
    mode=WindowMode.SINGLE_WINDOW,
    title="Relative Rotation Graph",
    width=1400,
    height=900,
)


def _get_cv(components, cid, default=None):
    entry = components.get(cid)
    if entry and isinstance(entry, dict):
        return entry.get("value", default)
    return default


def _read_params(components):
    return {
        "symbols": _get_cv(components, "symbols", ",".join(module.SPDRS)),
        "benchmark": _get_cv(components, "benchmark", "SPY"),
        "study": _get_cv(components, "study", "price"),
        "long_period": int(_get_cv(components, "long_period", 252)),
        "short_period": int(_get_cv(components, "short_period", 21)),
        "window": int(_get_cv(components, "window", 21)),
        "trading_periods": int(_get_cv(components, "trading_periods", 252)),
        "date": _get_cv(components, "date", datetime.today().strftime("%Y-%m-%d")),
        "show_tails": _show_tails,
        "tail_periods": int(_get_cv(components, "tail_periods", 30)),
        "tail_interval": _get_cv(components, "tail_interval", "week"),
    }


def _needs_fetch(params):
    if rrg_data is None:
        return True
    for k in _FETCH_KEYS:
        if params.get(k) != _last_fetch_params.get(k):
            return True
    return False


def _apply_dark_theme(fig):
    fig.update_layout(
        autosize=True,
        height=None,
        margin=dict(l=60, r=60, b=50, t=60, pad=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(30,30,50,1)",
        font=dict(color="#e0e0e0"),
        title_font=dict(color="#e0e0e0"),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.1)",
            zerolinecolor="rgba(255,255,255,0.4)",
            linecolor="rgba(255,255,255,0.3)",
            title_font=dict(color="#ccc"),
            tickfont=dict(color="#aaa"),
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.1)",
            zerolinecolor="rgba(255,255,255,0.4)",
            linecolor="rgba(255,255,255,0.3)",
            title_font=dict(color="#ccc"),
            tickfont=dict(color="#aaa"),
        ),
    )
    cmap = {"Leading": "#66ff66", "Weakening": "#ffcc00",
            "Lagging": "#ff6666", "Improving": "#6699ff"}
    for ann in fig.layout.annotations:
        if ann.text in cmap:
            ann.font.color = cmap[ann.text]
    for shape in fig.layout.shapes:
        if shape.line and shape.line.color == "Black":
            shape.line.color = "rgba(255,255,255,0.4)"
    qc = {"lightgreen": "rgba(0,180,0,0.15)", "lightblue": "rgba(0,100,255,0.15)",
          "lightpink": "rgba(255,50,50,0.15)", "lightyellow": "rgba(255,200,0,0.12)"}
    for shape in fig.layout.shapes:
        if shape.fillcolor in qc:
            shape.fillcolor = qc[shape.fillcolor]
            shape.opacity = 1.0
    # Style play button and slider for dark theme
    if fig.layout.updatemenus:
        for menu in fig.layout.updatemenus:
            menu.bgcolor = "#333"
            menu.bordercolor = "#666"
            menu.borderwidth = 1
            menu.font = dict(color="#fff", size=13)
    if fig.layout.sliders:
        for slider in fig.layout.sliders:
            if slider.currentvalue:
                slider.currentvalue.font = dict(size=14, color="#ccc")
            slider.font = dict(color="#aaa", size=11)
            slider.activebgcolor = "#666"
            slider.bgcolor = "#333"
            slider.bordercolor = "#555"
            slider.tickcolor = "#666"


def _apply_light_theme(fig):
    fig.update_layout(
        autosize=True,
        height=None,
        margin=dict(l=60, r=60, b=50, t=60, pad=0),
        paper_bgcolor="rgba(255,255,255,1)",
        plot_bgcolor="rgba(255,255,255,1)",
        font=dict(color="#333"),
        title_font=dict(color="#222"),
        xaxis=dict(
            gridcolor="rgba(0,0,0,0.1)",
            zerolinecolor="rgba(0,0,0,0.4)",
            linecolor="rgba(0,0,0,0.3)",
            title_font=dict(color="#444"),
            tickfont=dict(color="#555"),
        ),
        yaxis=dict(
            gridcolor="rgba(0,0,0,0.1)",
            zerolinecolor="rgba(0,0,0,0.4)",
            linecolor="rgba(0,0,0,0.3)",
            title_font=dict(color="#444"),
            tickfont=dict(color="#555"),
        ),
    )
    cmap = {"Leading": "darkgreen", "Weakening": "goldenrod",
            "Lagging": "red", "Improving": "blue"}
    for ann in fig.layout.annotations:
        if ann.text in cmap:
            ann.font.color = cmap[ann.text]
    # Keep original shape colors for light mode (already light-friendly)
    if fig.layout.updatemenus:
        for menu in fig.layout.updatemenus:
            menu.bgcolor = "#eee"
            menu.bordercolor = "#ccc"
            menu.borderwidth = 1
            menu.font = dict(color="#333", size=13)
    if fig.layout.sliders:
        for slider in fig.layout.sliders:
            if slider.currentvalue:
                slider.currentvalue.font = dict(size=14, color="#333")
            slider.font = dict(color="#555", size=11)
            slider.activebgcolor = "#aaa"
            slider.bgcolor = "#ddd"
            slider.bordercolor = "#bbb"
            slider.tickcolor = "#ccc"


def _apply_theme(fig):
    if _dark_mode:
        _apply_dark_theme(fig)
    else:
        _apply_light_theme(fig)


def _fetch_data(params):
    global rrg_data, _last_fetch_params
    symbols = [s.strip().upper() for s in params["symbols"].split(",") if s.strip()]
    benchmark = params["benchmark"].strip().upper()
    loop = asyncio.new_event_loop()
    rrg_data = loop.run_until_complete(
        module.create(
            symbols=symbols, benchmark=benchmark,
            study=params["study"].lower(),
            date=pd.to_datetime(params["date"]),
            long_period=params["long_period"],
            short_period=params["short_period"],
            window=params["window"],
            trading_periods=params["trading_periods"],
            tail_periods=params["tail_periods"],
            tail_interval=params["tail_interval"].lower(),
        )
    )
    loop.close()
    _last_fetch_params = {k: params[k] for k in _FETCH_KEYS}


def _build_figure(params):
    """Build Plotly figure from cached rrg_data. Returns the figure."""
    from openbb_charting.charts.relative_rotation import (
        create_rrg_with_tails,
        create_rrg_without_tails,
    )

    results = rrg_data.results
    ratios = basemodel_to_df(results.rs_ratios).set_index("date")
    ratios.index = pd.to_datetime(ratios.index)
    momentum = basemodel_to_df(results.rs_momentum).set_index("date")
    momentum.index = pd.to_datetime(momentum.index)

    bm = params["benchmark"].strip().upper()
    study = params["study"].lower()
    date = pd.to_datetime(params["date"])
    tails = bool(params["show_tails"])

    if tails:
        fig = create_rrg_with_tails(
            ratios_data=ratios, momentum_data=momentum,
            benchmark_symbol=bm, study=study,
            tail_periods=params["tail_periods"],
            tail_interval=params["tail_interval"].lower(),
        )
    else:
        fig = create_rrg_without_tails(
            ratios_data=ratios, momentum_data=momentum,
            benchmark_symbol=bm, study=study, date=date,
        )
    _apply_theme(fig)
    return fig


def _full_render(params):
    """Full window rebuild — used on first fetch or when sidebar structure changes."""
    global _chart_shown, _last_rebuild
    if rrg_data is None:
        return
    fig = _build_figure(params)

    _last_rebuild = _time.monotonic()
    _chart_shown = True

    # Use fig.to_html() to get complete Plotly HTML with frames/animation
    # This bypasses PyWry's build_plotly_init_script which drops frames
    plot_html = fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={
            "scrollZoom": True,
            "displaylogo": False,
            "responsive": True,
            "displayModeBar": False,
        },
    )
    # Force the plotly div to fill the container
    wrapped = (
        plot_html
        + '<style>'
        + '.pywry-scroll-container{overflow:hidden!important;padding:0!important;}'
        + '.pywry-scroll-container>div{width:100%!important;height:100%!important;}'
        + '.js-plotly-plot{width:100%!important;height:100%!important;}'
        + '.plot-container{width:100%!important;height:100%!important;}'
        + '.svg-container{width:100%!important;height:100%!important;}'
        + '</style>'
        + '<script>'
        + 'setTimeout(function(){'
        + '  var gd=document.querySelector(".js-plotly-plot");'
        + '  if(gd&&window.Plotly)Plotly.Plots.resize(gd);'
        + '},200);'
        + 'window.addEventListener("resize",function(){'
        + '  var gd=document.querySelector(".js-plotly-plot");'
        + '  if(gd&&window.Plotly)Plotly.Plots.resize(gd);'
        + '});'
        + '</script>'
    )
    app.show(
        HtmlContent(html=wrapped, inline_css=SIDEBAR_CSS),
        title="Relative Rotation Graph",
        include_plotly=True,
        toolbars=build_toolbars(),
        callbacks=CALLBACKS,
    )


def _update_chart(params):
    """Update chart in-place without rebuilding toolbar. No infinite loop."""
    if rrg_data is None:
        return
    fig = _build_figure(params)
    fig_dict = fig.to_plotly_json()
    # Send the figure update via event — no toolbar rebuild
    app.emit("plotly:update-figure", {
        "data": fig_dict.get("data", []),
        "layout": fig_dict.get("layout", {}),
        "frames": fig_dict.get("frames", []),
    })


def _show_initial():
    """Show the initial landing page."""
    global _last_rebuild
    _last_rebuild = _time.monotonic()
    app.show(
        HtmlContent(
            html="""
            <div style="display:flex;align-items:center;justify-content:center;height:100%;
                        font-family:system-ui;color:#ccc;flex-direction:column;gap:16px;">
                <h1 style="margin:0;font-size:2em;">Relative Rotation Graph</h1>
                <p style="margin:0;font-size:1.1em;opacity:0.7;">
                    Configure parameters in the sidebar and click <b>Fetch Data</b>.
                </p>
                <p style="margin:0;font-size:0.85em;opacity:0.5;">
                    Powered by OpenBB Platform &bull; PyWry Desktop
                </p>
            </div>
            """,
            inline_css=SIDEBAR_CSS,
        ),
        title="Relative Rotation Graph",
        toolbars=build_toolbars(),
        callbacks=CALLBACKS,
    )


# ── Callbacks ─────────────────────────────────────────────────────────
def on_submit(data, event_type, label):
    app.emit("toolbar:request-state", {"context": "submit"})


def on_study_change(data, event_type, label):
    """Study changed — rebuild sidebar (volatility fields appear/disappear)."""
    global _current_study
    if _time.monotonic() - _last_rebuild < 1.0:
        return
    val = data.get("value", "price") if isinstance(data, dict) else "price"
    if val.lower() != _current_study:
        _current_study = val.lower()
        if _chart_shown and rrg_data is not None:
            # Need full rebuild to change sidebar, then re-render chart
            app.emit("toolbar:request-state", {"context": "study-rebuild"})
        else:
            _show_initial()


def on_theme_toggle(data, event_type, label):
    global _dark_mode
    if _time.monotonic() - _last_rebuild < 1.0:
        return
    new_val = data.get("value", True) if isinstance(data, dict) else True
    if new_val != _dark_mode:
        _dark_mode = new_val
        app.emit("pywry:update-theme", {"theme": "dark" if _dark_mode else "light"})
        if _chart_shown and rrg_data is not None:
            app.emit("toolbar:request-state", {"context": "render"})


def on_show_tails_change(data, event_type, label):
    global _show_tails
    if _time.monotonic() - _last_rebuild < 1.0:
        return
    new_val = data.get("value", False) if isinstance(data, dict) else False
    if new_val != _show_tails:
        _show_tails = new_val
        if _chart_shown and rrg_data is not None:
            app.emit("toolbar:request-state", {"context": "render"})


def on_toolbar_state(data, event_type, label):
    components = data.get("components", {}) if isinstance(data, dict) else {}
    context = data.get("context", "submit") if isinstance(data, dict) else "submit"
    params = _read_params(components)
    try:
        if context == "submit":
            if _needs_fetch(params):
                app.alert("Fetching data...", alert_type="info", duration=3000)
                _fetch_data(params)
                # First fetch or sidebar-affecting change: full rebuild
                _full_render(params)
                app.alert("Data loaded!", alert_type="success", duration=1500)
            else:
                # Same data params, just re-render (date/tails changed)
                _update_chart(params)
                app.alert("Chart updated!", alert_type="success", duration=1000)
        elif context == "study-rebuild":
            # Study changed — need full rebuild for sidebar, then chart
            _full_render(params)
        elif context == "render":
            # Tails toggled — update chart in-place
            _update_chart(params)
    except Exception as exc:
        app.alert(f"Error: {exc}", alert_type="error", duration=5000)


CALLBACKS = {
    "rrg:submit": on_submit,
    "study:change": on_study_change,
    "theme:toggle": on_theme_toggle,
    "showtails:change": on_show_tails_change,
    "toolbar:state-response": on_toolbar_state,
}


# ── Dynamic Sidebar Builder ──────────────────────────────────────────
def _section(title):
    return Div(content=title, class_name="sidebar-heading")


def _rule():
    return Div(content="<hr class='sidebar-rule'>")


def build_toolbars():
    items = [
        Toggle(label="Dark Mode", event="theme:toggle",
               component_id="dark_mode", value=_dark_mode),
        _rule(),
        TextInput(label="Symbols", event="symbols:change",
                  component_id="symbols", value=",".join(module.SPDRS)),
        TextInput(label="Benchmark", event="benchmark:change",
                  component_id="benchmark", value="SPY"),
        DateInput(label="Target End Date", event="date:change",
                  component_id="date", value=datetime.today().strftime("%Y-%m-%d")),
        _rule(),
        Select(
            label="Study", event="study:change", component_id="study",
            options=[Option(label="Price", value="price"),
                     Option(label="Volume", value="volume"),
                     Option(label="Volatility", value="volatility")],
            selected=_current_study,
        ),
    ]

    if _current_study == "volatility":
        items += [
            _section("Volatility Annualization"),
            NumberInput(label="Rolling Window", event="window:change",
                        component_id="window", value=21, min=1, max=500),
            NumberInput(label="Periods Per Year", event="tradingperiods:change",
                        component_id="trading_periods", value=252, min=1, max=1000),
        ]

    items += [
        _rule(),
        _section("Momentum Periods"),
        Div(
            style="display:flex;flex-direction:row;gap:8px;width:100%;",
            children=[
                NumberInput(label="Long", event="longperiod:change",
                            component_id="long_period", value=252, min=1, max=1000),
                NumberInput(label="Short", event="shortperiod:change",
                            component_id="short_period", value=21, min=1, max=500),
            ],
        ),
        _rule(),
        Button(label="Fetch Data", event="rrg:submit", variant="primary",
               size="lg", style="width:100%;margin-top:4px;margin-bottom:4px;"),
        _rule(),
        Checkbox(label="Show Tails", event="showtails:change",
                 component_id="show_tails", value=_show_tails),
        Div(
            style="display:flex;flex-direction:row;gap:8px;width:100%;",
            children=[
                NumberInput(label="Periods", event="tailperiods:change",
                            component_id="tail_periods", value=30, min=1, max=200),
                Select(
                    label="Interval", event="tailinterval:change",
                    component_id="tail_interval",
                    options=[Option(label="Week", value="week"),
                             Option(label="Month", value="month")],
                    selected="week",
                ),
            ],
        ),
    ]

    return [Toolbar(position="left", collapsible=True, items=items)]


# ── Launch ────────────────────────────────────────────────────────────
_show_initial()
app.block()
