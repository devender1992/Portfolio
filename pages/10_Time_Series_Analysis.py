import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Set page config for this specific page
st.set_page_config(
    page_title="Time Series Analysis Case Study",
    page_icon="⏳",
    layout="wide"
)

st.title("⏳ Project: Time Series Analysis Case Study")
st.markdown("""
This section demonstrates **Time Series Analysis**, a specialized area of data analysis
focused on understanding and forecasting data points indexed in time order. Time series
data is prevalent in finance, economics, weather forecasting, sales, and many other domains.

We will generate synthetic daily sales data and apply common techniques to uncover
trends, seasonality, and make basic forecasts.
""")

st.subheader("1. Data Generation & Overview (Synthetic Daily Sales Data)")
st.write("We'll generate a synthetic dataset of daily sales values with a clear trend, weekly seasonality, and some random noise.")

# User inputs for data generation
st.sidebar.header("Time Series Data Parameters")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime('2023-01-01'))
num_days = st.sidebar.slider("Number of Days", 100, 730, 365, 50)
base_sales = st.sidebar.slider("Base Daily Sales", 50, 200, 100)
trend_per_day = st.sidebar.slider("Trend per Day", 0.0, 1.0, 0.1, 0.01)
seasonal_strength = st.sidebar.slider("Weekly Seasonal Strength", 0.0, 0.5, 0.2, 0.05)
noise_level = st.sidebar.slider("Noise Level", 0.0, 0.2, 0.05, 0.01)


@st.cache_data
def generate_time_series_data(start_date, num_days, base_sales, trend_per_day, seasonal_strength, noise_level):
    """Generates synthetic daily sales time series data."""
    np.random.seed(42)

    dates = pd.date_range(start=start_date, periods=num_days, freq='D')
    df = pd.DataFrame({'Date': dates})

    # Base sales with trend
    df['Sales'] = base_sales + (df.index * trend_per_day)

    # Add weekly seasonality
    # Day of week: Monday=0, Sunday=6
    weekly_pattern = np.array([0.9, 1.0, 1.1, 1.05, 1.15, 1.25, 0.85]) # Example pattern
    df['Seasonal_Factor'] = weekly_pattern[df.Date.dt.dayofweek]
    df['Sales'] = df['Sales'] * (1 + seasonal_strength * (df['Seasonal_Factor'] - 1))

    # Add random noise
    df['Sales'] = df['Sales'] + np.random.normal(0, noise_level * base_sales, num_days)

    df['Sales'] = np.maximum(df['Sales'], 1).round(2) # Ensure positive sales
    return df

sales_ts_df = generate_time_series_data(start_date, num_days, base_sales, trend_per_day, seasonal_strength, noise_level)

st.write("#### Sample of Generated Sales Data:")
st.dataframe(sales_ts_df.head())
st.write(f"Total data points: {sales_ts_df.shape[0]}")

st.write("#### Raw Time Series Plot: Daily Sales")
fig_raw_ts = px.line(sales_ts_df, x='Date', y='Sales', title='Raw Daily Sales Data Over Time',
                     template='plotly_white')
fig_raw_ts.update_xaxes(rangeslider_visible=True)
st.plotly_chart(fig_raw_ts, use_container_width=True, key='raw_time_series_plot')
st.info("This plot shows the overall pattern of sales, including any visible trends or repetitive fluctuations.")

st.markdown("---")
st.subheader("2. Trend Analysis: Moving Averages")
st.markdown("""
A **moving average (rolling mean)** is a popular technique to smooth out short-term fluctuations
and highlight longer-term trends or cycles in time series data.
""")

window_size = st.slider("Select Moving Average Window Size (Days)", 3, 30, 7)
sales_ts_df[f'Moving_Avg_{window_size}_Days'] = sales_ts_df['Sales'].rolling(window=window_size).mean()

st.write(f"#### Daily Sales with {window_size}-Day Moving Average")
fig_ma = px.line(sales_ts_df, x='Date', y=['Sales', f'Moving_Avg_{window_size}_Days'],
                 title=f'Daily Sales vs. {window_size}-Day Moving Average',
                 template='plotly_white')
fig_ma.update_xaxes(rangeslider_visible=True)
fig_ma.data[1].line.color = 'red' # Make moving average line stand out
st.plotly_chart(fig_ma, use_container_width=True, key='moving_average_plot')
st.info("The moving average smooths the data, making the underlying trend more apparent.")

st.markdown("---")
st.subheader("3. Seasonality Analysis")
st.markdown("""
Seasonality refers to periodic fluctuations in a time series that occur regularly and are
of constant length. We'll examine weekly seasonality in our sales data.
""")

st.write("#### Average Sales by Day of Week")
sales_ts_df['DayOfWeek'] = sales_ts_df['Date'].dt.day_name()
# Ensure consistent order of days of week
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
sales_ts_df['DayOfWeek'] = pd.Categorical(sales_ts_df['DayOfWeek'], categories=day_order, ordered=True)

avg_sales_by_day = sales_ts_df.groupby('DayOfWeek')['Sales'].mean().reset_index()

fig_dow = px.bar(avg_sales_by_day, x='DayOfWeek', y='Sales',
                 title='Average Sales by Day of Week',
                 color='Sales', color_continuous_scale=px.colors.sequential.Plasma,
                 template='plotly_white')
st.plotly_chart(fig_dow, use_container_width=True, key='sales_by_day_of_week')
st.info("This bar chart clearly shows the weekly sales pattern, indicating seasonality.")

st.markdown("---")
st.subheader("4. Time Series Decomposition")
st.markdown("""
Time series decomposition breaks down a time series into its fundamental components:
**Trend**, **Seasonality**, and **Residuals (or Noise)**. This helps in understanding
the underlying structure of the data.
""")
# Set Date as index for decomposition
ts_data = sales_ts_df.set_index('Date')['Sales'].asfreq('D')

# Perform additive decomposition (assuming seasonality magnitude is constant over time)
# For multiplicative, use model='multiplicative'
decomposition = seasonal_decompose(ts_data, model='additive', period=7) # Period is 7 for weekly seasonality

st.write("#### Decomposed Time Series Components:")
fig_decomp = decomposition.plot()
fig_decomp.set_size_inches(10, 8) # Adjust size for better readability in Streamlit
plt.tight_layout()
st.pyplot(fig_decomp) # No 'key' needed for matplotlib figures
plt.close(fig_decomp) # Close plot to prevent memory issues

st.info("""
* **Trend:** The long-term direction of the series.
* **Seasonal:** The repeating pattern that occurs over a fixed period (e.g., weekly, monthly, yearly).
* **Residual:** The random variation or noise left after removing the trend and seasonal components.
""")

st.markdown("---")
st.subheader("5. Simple Forecasting Example (Naive Approach)")
st.markdown("""
Forecasting involves predicting future values based on historical data. A simple
naive forecasting method predicts the next value to be the same as the last observed value
or the average of previous values. For this demo, we'll use a simple moving average forecast.
""")

# Define forecast horizon
forecast_horizon = st.slider("Select Forecast Horizon (Days)", 1, 30, 7)

# Simple forecast: Use the last observed moving average as forecast for next 'horizon' days
last_ma_value = sales_ts_df[f'Moving_Avg_{window_size}_Days'].iloc[-1]
last_date = sales_ts_df['Date'].iloc[-1]

forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon, freq='D')
forecast_df = pd.DataFrame({'Date': forecast_dates, 'Sales': last_ma_value})
forecast_df['Type'] = 'Forecast'

# Prepare actual data for plotting
actual_data_for_plot = sales_ts_df[['Date', 'Sales']].copy()
actual_data_for_plot['Type'] = 'Actual'

# Combine actual and forecast for plotting
combined_plot_df = pd.concat([actual_data_for_plot, forecast_df])

fig_forecast = px.line(combined_plot_df, x='Date', y='Sales', color='Type',
                       title=f'Simple Moving Average Forecast ({forecast_horizon} Days)',
                       color_discrete_map={'Actual': 'blue', 'Forecast': 'red'},
                       template='plotly_white')
fig_forecast.update_xaxes(rangeslider_visible=True)
st.plotly_chart(fig_forecast, use_container_width=True, key='simple_forecast_plot')
st.info(f"""
This is a very basic forecast using the last {window_size}-day moving average.
More advanced forecasting models (e.g., ARIMA, Prophet, Neural Networks) are used for
complex time series with multiple seasonality and external factors.
""")

st.markdown("---")
st.subheader("Key Takeaways from Time Series Analysis:")
st.markdown("""
* **Temporal Dependency:** Time series data has a sequential dependency, meaning past values influence future ones.
* **Components:** Understanding trend, seasonality, and residuals is key to effective time series analysis.
* **Forecasting:** Predicting future values is a core application, enabling strategic planning and resource allocation.
* **Applications:** Sales forecasting, stock market prediction, weather forecasting, demand planning, and many more.
* **Complexity:** Real-world time series can be complex, involving multiple seasonality, structural breaks, and external variables, requiring advanced modeling techniques.
""")
