# Relative Rotation Graph Dashboard

This dashboard is powered by the OpenBB Platform. More information can be found here: https://docs.openbb.co/platform

A Relative Rotation Graph is a study of the Relative Strength Ratio vs. Relative Strength Momentum against a
benchmark. They are lagging indicators and are typically used for comparing sector or index
cconstituents.

The Relative Strength Ratio is the price (volume or realized volatility) of the asset divided by the
benchmark.

The Relative Strength (RS) Momentum is the momentum of the Relative Strength Ratio.
In this application, momentum is calculated as the trailing 12-month minus 1-month return.
The default values for long and short periods are 252 and 21, which is the number of trading
days in a year and a month, respectively. These values can be changed in the sidebar.

Realized volatility is calculated as the annualized standard deviation over a trailing 1-month period.
The number of trading days per year and the rolling window can be adjusted in the sidebar.
            
All calculations use daily close levels from the source selected in the sidebar. It should not be assumed
that volume represents 100% market coverage.

All values are normalized using the Z-Score Standardization method.

This dashboard is for demonstration purposes only and
should not be used to make inferences or investment decisions.
